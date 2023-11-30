import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(12,9)})
import NFL_Functions as NFL


# read the data
data = pd.read_csv('nfl_team_stats_2002-2019.csv')
data = NFL.prep_team_stats_df(data)

# create the data frame for power ranking predicitons
df, team_names = NFL.prep_power_rank_df(data)



#### Initialize Variables
# Parameter Settings
Std = 7 # stdev of PwrX
StdPrior = 2 #Initial Stdev of prior
dm = 0.1
ds = 0.2
dx = dm*ds # used for numerical integration in denominator of Bayes them
x = np.arange(-30,30,dm)
mn, stdv = np.meshgrid(np.arange(-30.,30.,dm), np.arange(5., 20.,ds))

# Initialize the Power Ranking
num_teams = np.max(df['home_idx'])+1
num_days = np.max(df['day_number'])+1
Current_PowerRank = np.zeros([num_teams, 2]) # mean and standard deviation for E[ScoreDiff] per team
Current_PowerRank[:,1] = Std
Current_PowerRankPrior = np.ones([num_teams, np.shape(mn)[0], np.shape(mn)[1]])
# This is the power rank ( E[ScoreDiff] ) for each team and each day that games are played
PowerRank = np.ones([num_days,num_teams])
# Record accuracy over time
PowerRank_correct_per_day = np.zeros(num_days)
PowerRank_games_per_day = np.zeros(num_days)

# Create the array of inputs from games played
PowerRank_inputs = df[['home_idx','away_idx','score_diff','day_number']].to_numpy()
Predicted_score_diff = np.zeros(PowerRank_inputs.shape[0])
Predicted_score_diff_zscore = np.zeros(PowerRank_inputs.shape[0])
Predicted_score_diff_err = np.zeros(PowerRank_inputs.shape[0])

idx = 0
for r in PowerRank_inputs:
    # Compute the home and away win probabilities based on their power rankings
    # Assuming a logistic
    Home_Team_Pwr = copy.deepcopy(Current_PowerRank[r[0],:])
    Home_Team_Pwr_prior = copy.deepcopy(Current_PowerRankPrior[r[0],:,:])
    Away_Team_Pwr = copy.deepcopy(Current_PowerRank[r[1],:])
    Away_Team_Pwr_prior = np.flip(copy.deepcopy(Current_PowerRankPrior[r[1],:,:]),1)
    # reverse the sign on the away team (so the away team tries to get the score low in score_diff)
    Away_Team_Pwr[0] = (-1)*Away_Team_Pwr[0]
    SDiff = r[2]-2.4

    #print('Home Pwr Mean: '+str(Home_Team_Pwr[0])+' Stdev: '+str(Home_Team_Pwr[1]))
    #print('Away Pwr Mean: '+str(Away_Team_Pwr[0])+' Stdev: '+str(Away_Team_Pwr[1]))

    #### Predict Step: Compute the predicted score distribution
    SDiff_predicted = NFL.predict_outcome(Home_Team_Pwr, Away_Team_Pwr)
    #print('Predicted Score Diff Mean: '+str(SDiff_predicted[0])+' Stdev: '+str(SDiff_predicted[1]))

    # Record the results
    Predicted_score_diff[idx] = SDiff_predicted[0]
    PowerRank_games_per_day[r[3]] = PowerRank_games_per_day[r[3]] + 1
    PowerRank_correct_per_day[r[3]] = PowerRank_correct_per_day[r[3]] + (SDiff_predicted[0]*SDiff > 0)*1
    Predicted_score_diff_zscore[idx] = (SDiff_predicted[0]-SDiff)/SDiff_predicted[1]
    Predicted_score_diff_err[idx]= SDiff_predicted[0]-SDiff

    #### Estimate Step: use Bayes Thm and the game outcome to compute new power distributions and priors
    Home_Team_Pwr_new, Home_Team_Pwr_prior_new, Away_Team_Pwr_new, Away_Team_Pwr_prior_new = NFL.update_power_distributions(
        Home_Team_Pwr, Home_Team_Pwr_prior, Away_Team_Pwr, Away_Team_Pwr_prior, SDiff, dx, mn, stdv)

    Current_PowerRank[r[0], 0] = Home_Team_Pwr_new[0]
    Current_PowerRank[r[0], 1] = Home_Team_Pwr_new[1]
    Current_PowerRank[r[1], 0] = (-1)*Away_Team_Pwr_new[0]
    Current_PowerRank[r[1], 1] = Away_Team_Pwr_new[1]
    Current_PowerRankPrior[r[0], :, :] = Home_Team_Pwr_prior_new
    Current_PowerRankPrior[r[1], :, :] = np.flip(Away_Team_Pwr_prior_new, 1)
    idx = idx + 1
    print(idx)
    if idx in [1,1000,1001,1002,1003,1004]:
        NFL.plot_update_results(
            Home_Team_Pwr, Home_Team_Pwr_new, team_names[r[0]],
            Away_Team_Pwr, Away_Team_Pwr_new, team_names[r[1]], SDiff, SDiff_predicted, x)
        NFL.plot_priors(mn, stdv, Home_Team_Pwr_prior, team_names[r[0]], Away_Team_Pwr_prior, team_names[r[1]])
        NFL.plot_priors(mn, stdv, Home_Team_Pwr_prior_new, team_names[r[0]]+' (updated)', Away_Team_Pwr_prior_new, team_names[r[1]]+' (updated)')
        if SDiff_predicted[0] > 0:
            print('Prediction: Home Team by '+str(SDiff_predicted[0]))
        else:
            print('Prediction: Away Team by '+str((-1)*SDiff_predicted[0]))
        if SDiff > 0:
            print('Game Result: Home Team by '+str(SDiff))
        else:
            print('Game Result: Away Team by '+str((-1)*SDiff))

print('accuracy: '+str(np.sum(PowerRank_correct_per_day)/np.sum(PowerRank_games_per_day)))
print('RMSE: '+str(np.sqrt(np.mean(Predicted_score_diff_err[500:]**2))))
print('done')



NFL.plot_priors(mn, stdv, Home_Team_Pwr_prior_new, team_names[r[0]], Away_Team_Pwr_prior_new, team_names[r[1]])