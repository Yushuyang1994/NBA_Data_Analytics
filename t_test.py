import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)

# Hypothesis Test1
career_stats = pd.read_csv('player_career_stats_cleaned.csv')

pros = career_stats[career_stats['PTS'] > 10]
norms = career_stats[career_stats['PTS'] <= 10]

pros_fg_percentage = np.asarray(pros['FG%'])
norms_fg_percentage = np.asarray(norms['FG%'])

ttest = ttest_ind(pros_fg_percentage, norms_fg_percentage)
print(ttest)

# Hypothesis Test 2
player_stats = pd.read_csv('player_stats_cleaned.csv')

df = player_stats.groupby(['Name'], sort=False, as_index=False).agg(lambda x: x.value_counts().index[0])
df = df[['Name', 'Pos']]

career_stats_with_pos = pd.merge(career_stats, df, sort=False)

print(career_stats_with_pos.head())
print(career_stats_with_pos['Pos'].value_counts())
c = np.asarray(career_stats_with_pos[career_stats_with_pos['Pos']=='C']['PTS'])
sf = np.asarray(career_stats_with_pos[career_stats_with_pos['Pos']=='SF']['PTS'])
sg = np.asarray(career_stats_with_pos[career_stats_with_pos['Pos']=='SG']['PTS'])
pg = np.asarray(career_stats_with_pos[career_stats_with_pos['Pos']=='PG']['PTS'])
pf = np.asarray(career_stats_with_pos[career_stats_with_pos['Pos']=='PF']['PTS'])
pos = [c, sf, sg, pg, pf]
names = ['c', 'sf', 'sg', 'pg', 'pf']

for i in range(5):
    for j in range(5):
        ttest = ttest_ind(pos[i], pos[j])
        print(names[i], names[j], ttest)


print(career_stats.head())
y = np.asarray(career_stats['PTS'])
x = np.asarray(career_stats['FG%'])
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())

