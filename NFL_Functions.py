import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

def mult_Guassians(m1,s1,m2,s2):
    m = (s2**2*m1 + s1**2*m2)/(s1**2 + s2**2)
    s = np.sqrt(1/(s1**(-2) + s2**(-2)))
    return np.array([m,s])

def predict_outcome(PwrA, PwrB):
    return mult_Guassians(PwrA[0],PwrA[1],PwrB[0],PwrB[1])

def update_power_distributions(PwrA, PwrA_prior, PwrB, PwrB_prior, SDiff, dx, mn, stdv):

    # Prediction Step for PwrA
    m1 = mn  # we are computing the posterior probability distribution for the mean of PwrA over all values of x
    s1 = stdv
    m2 = PwrB[0]
    s2 = PwrB[1]
    m = (s2**2 * m1 + s1**2 * m2) / (s1**2 + s2**2)
    s = np.sqrt(1 / (s1**(-2) + s2**(-2)))
    likelihood_teamA_mean = norm.pdf(SDiff, loc=m, scale=s)
    # Use Bayes Thm to predict the new distribution
    PwrA_prior_new = likelihood_teamA_mean * PwrA_prior
    PwrA_prior_new = PwrA_prior_new / (dx*np.sum(PwrA_prior_new))
    # update the mean of PwrA
    index_max = np.unravel_index(PwrA_prior_new.argmax(), PwrA_prior_new.shape)
    PwrA_new = [mn[index_max],stdv[index_max]]

    # Prediction Step for PwrB
    m1 = PwrA[0]
    s1 = PwrA[1]
    m2 = mn # we are computing the posterior probability distribution for the mean of PwrB over all values of x
    s2 = stdv
    m = (s2 ** 2 * m1 + s1 ** 2 * m2) / (s1 ** 2 + s2 ** 2)
    s = np.sqrt(1 / (s1 ** (-2) + s2 ** (-2)))
    likelihood_teamB_mean = norm.pdf(SDiff, loc=m, scale=s)
    PwrB_prior_new = likelihood_teamB_mean * PwrB_prior
    PwrB_prior_new = PwrB_prior_new / (0.1*np.sum(PwrB_prior_new))
    index_max = np.unravel_index(PwrB_prior_new.argmax(), PwrB_prior_new.shape)
    PwrB_new = [mn[index_max],stdv[index_max]]

    return PwrA_new, PwrA_prior_new, PwrB_new, PwrB_prior_new

def plot_prediction(Home_Team_Pwr, Away_Team_Pwr, Home_Team_Pwr_prior, Away_Team_Pwr_prior, SDiff_predicted, x):
    # plot setup
    pA = norm.pdf(x, loc=Home_Team_Pwr[0], scale=Home_Team_Pwr[1])
    pB = norm.pdf(x, loc=Away_Team_Pwr[0], scale=Away_Team_Pwr[1])
    pPred = norm.pdf(x, loc=SDiff_predicted[0], scale=SDiff_predicted[1])

    # plot the team capabilities with prediction
    fig1 = plt.figure(1)
    plt.plot(x, pA, color='b', label='Team A SDiff probability')
    plt.plot(x, pB, color='g', label='Team B SDiff probability')
    plt.plot(x, pPred, color='r', label='Game Outcome SDiff probability')
    plt.vlines(SDiff_predicted[0], ymin=-0, ymax=0.2 * np.max(pPred), color='r')
    plt.fill_between(x, Home_Team_Pwr_prior, 0, facecolor="b", alpha=0.1)
    plt.fill_between(x, Away_Team_Pwr_prior, 0, facecolor="g", alpha=0.1)
    plt.xlabel('Score Difference')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def plot_update_results(PwrA, PwrA_new, team_A_name, PwrB, PwrB_new, team_B_name, SDiff, SDiff_predicted, x):

    # plot setup
    pA = norm.pdf(x, loc=PwrA[0], scale=PwrA[1])
    pB = norm.pdf(x, loc=PwrB[0], scale=PwrB[1])
    pPred = norm.pdf(x, loc=SDiff_predicted[0], scale=SDiff_predicted[1])

    # plot the team capabilities with prediction
    fig1 = plt.figure(1)
    plt.plot(x, pA, color='b', label=team_A_name+' (Home) SDiff probability')
    plt.plot(x, pB, color='g', label=team_B_name+' (Away) SDiff probability')
    plt.plot(x, pPred, color='r', label='Game Outcome SDiff probability')
    plt.vlines(SDiff_predicted[0], ymin=-0, ymax=0.2 * np.max(pPred), color='r')
    plt.xlabel('Score Difference')
    plt.ylabel('Probability Density')
    plt.legend()

    # plot setup
    pA = norm.pdf(x, loc=PwrA[0], scale=PwrA[1])
    pA_new = norm.pdf(x, loc=PwrA_new[0], scale=PwrA_new[1])
    pB = norm.pdf(x, loc=PwrB[0], scale=PwrB[1])
    pB_new = norm.pdf(x, loc=PwrB_new[0], scale=PwrB_new[1])
    pPred = norm.pdf(x, loc=SDiff_predicted[0], scale=SDiff_predicted[1])

    # plot the team capabilities with prediction
    fig2 = plt.figure(2)
    plt.plot(x, pA, 'b', label=team_A_name+' (Home) SDiff probability')
    plt.plot(x, pB, 'g', label=team_B_name+' (Away) SDiff probability')
    plt.plot(x, pPred, 'r', label='Game Outcome SDiff probability')
    plt.vlines(SDiff_predicted[0], ymin=-0, ymax=0.2 * np.max(pPred), color='r')
    plt.vlines(SDiff, ymin=-0, ymax=0.2 * np.max(pPred))
    plt.plot(x, pA_new, 'm--', label='Updated '+team_A_name+' SDiff probability')
    plt.plot(x, pB_new, 'c--', label='Updated '+team_B_name+' SDiff probability')
    plt.xlabel('Score Difference')
    plt.ylabel('Probability Density')
    plt.title('Predcicted Score Diff: '+str(SDiff_predicted[0])+' Game Scored Diff: '+str(SDiff))
    plt.legend()
    plt.show()

def prep_team_stats_df(data):
    data['score_diff'] = data['score_home'] - data['score_away']

    data['winner'] = (data['score_home'] - data['score_away']) > 0
    data['winner'] = data['winner'].astype(int)

    # split string info for third and fourth down attmepts to float succeses and rate
    third_downs_away = data['third_downs_away'].str.split('-', expand=True)
    data['third_downs_away'] = third_downs_away[0].to_numpy(dtype=int)
    data['third_downs_rate_away'] = np.nan_to_num(
        third_downs_away[0].to_numpy(dtype=float) / third_downs_away[1].to_numpy(dtype=float), nan=0.0)
    data['third_downs_attempts_away'] = third_downs_away[1].to_numpy(dtype=int)

    third_downs_home = data['third_downs_home'].str.split('-', expand=True)
    data['third_downs_home'] = third_downs_home[0].to_numpy(dtype=float)
    data['third_downs_rate_home'] = np.nan_to_num(
        third_downs_home[0].to_numpy(dtype=float) / third_downs_home[1].to_numpy(dtype=float), nan=0.0)
    data['third_downs_attempts_home'] = third_downs_home[1].to_numpy(dtype=float)

    # split string info for third and fourth down attmepts to float succeses and rate
    fourth_downs_away = data['fourth_downs_away'].str.split('-', expand=True)
    data['fourth_downs_away'] = fourth_downs_away[0].to_numpy(dtype=int)
    data['fourth_downs_rate_away'] = np.nan_to_num(
        fourth_downs_away[0].to_numpy(dtype=float) / fourth_downs_away[1].to_numpy(dtype=float), nan=0.0)
    data['fourth_downs_attempts_away'] = fourth_downs_away[1].to_numpy(dtype=int)

    fourth_downs_home = data['fourth_downs_home'].str.split('-', expand=True)
    data['fourth_downs_home'] = third_downs_home[0].to_numpy(dtype=float)
    data['fourth_downs_rate_home'] = np.nan_to_num(
        third_downs_home[0].to_numpy(dtype=float) / third_downs_home[1].to_numpy(dtype=float), nan=0.0)
    data['fourth_downs_attempts_home'] = third_downs_home[1].to_numpy(dtype=float)

    # split string info for completions to float succeses and rate
    comp_att_away = data['comp_att_away'].str.split('-', expand=True)
    data['complete_passes_away'] = comp_att_away[0].to_numpy(dtype=int)
    data['pass_completion_rate_away'] = np.nan_to_num(
        comp_att_away[0].to_numpy(dtype=float) / comp_att_away[1].to_numpy(dtype=float), nan=0.0)
    data['pass_attempt_away'] = comp_att_away[1].to_numpy(dtype=int)
    data.drop('comp_att_away', 1, inplace=True)

    comp_att_home = data['comp_att_home'].str.split('-', expand=True)
    data['complete_passes_home'] = comp_att_home[0].to_numpy(dtype=int)
    data['pass_completion_rate_home'] = np.nan_to_num(
        comp_att_home[0].to_numpy(dtype=float) / comp_att_home[1].to_numpy(dtype=float), nan=0.0)
    data['pass_attempt_home'] = comp_att_home[1].to_numpy(dtype=int)
    data.drop('comp_att_home', 1, inplace=True)

    # split string info for sacks to float succeses and rate
    sacks_away = data['sacks_away'].str.split('-', expand=True)
    data['sacks_away'] = sacks_away[0].to_numpy(dtype=int)
    data['pressures_away'] = sacks_away[1].to_numpy(dtype=int)

    sacks_home = data['sacks_home'].str.split('-', expand=True)
    data['sacks_home'] = sacks_home[0].to_numpy(dtype=int)
    data['pressures_home'] = sacks_home[1].to_numpy(dtype=int)

    # split string info for penalties to float succeses and rate
    penalties_away = data['penalties_away'].str.split('-', expand=True)
    data['penalties_away'] = np.nan_to_num(penalties_away[0].to_numpy(dtype=int), nan=0.0)
    data['penalties_yards_away'] = np.nan_to_num(penalties_away[1].to_numpy(dtype=int), nan=0.0)

    penalties_home = data['penalties_home'].str.split('-', expand=True)
    data['penalties_home'] = np.nan_to_num(penalties_home[0].to_numpy(dtype=int), nan=0.0)
    data['penalties_yards_home'] = np.nan_to_num(penalties_home[1].to_numpy(dtype=int), nan=0.0)

    # split string info for completions to float succeses and rate
    redzone_away = data['redzone_away'].str.split('-', expand=True)
    data['redzone_sucess_away'] = redzone_away[0].to_numpy(dtype=int)
    data['redzone_rate_away'] = np.nan_to_num(
        redzone_away[0].to_numpy(dtype=float) / redzone_away[1].to_numpy(dtype=float), nan=0.0)
    data['redzone_attempts_away'] = redzone_away[1].to_numpy(dtype=int)
    data.drop('redzone_away', 1, inplace=True)

    redzone_home = data['redzone_home'].str.split('-', expand=True)
    data['redzone_sucess_home'] = redzone_home[0].to_numpy(dtype=int)
    data['redzone_rate_home'] = np.nan_to_num(
        redzone_home[0].to_numpy(dtype=float) / redzone_home[1].to_numpy(dtype=float), nan=0.0)
    data['redzone_attempts_home'] = redzone_home[1].to_numpy(dtype=int)
    data.drop('redzone_home', 1, inplace=True)

    # split string info for completions to float succeses and rate
    possession_away = data['possession_away'].str.split(':', expand=True)
    data['possession_away'] = possession_away[0].to_numpy(dtype=float) * 60 + possession_away[1].to_numpy(dtype=float)

    possession_home = data['possession_home'].str.split(':', expand=True)
    data['possession_home'] = possession_home[0].to_numpy(dtype=float) * 60 + possession_home[1].to_numpy(dtype=float)

    day_numbers = pd.to_numeric((pd.to_datetime(data['date']) - pd.to_datetime(data['date'][0])).dt.days,
                  downcast='integer')
    years = np.ones(len(day_numbers)) * 2002
    for idx in range(1, len(day_numbers)):
        years[idx] = years[idx - 1] + ((day_numbers[idx] - day_numbers[idx - 1]) > 30) * 1
    data['season'] = years

    return data

def prep_power_rank_df(data):
    # dcreate the df data frame
    df = data[["date", "away", "home", "winner", "score_away", "score_home", "score_diff"]]
    df['day_number'] = pd.to_numeric((pd.to_datetime(df['date']) - pd.to_datetime(data['date'][0])).dt.days,
                                     downcast='integer')
    date_numbers = np.unique(df['day_number'].to_numpy())

    srt = np.argsort(date_numbers)
    date_number_copy = df['day_number'].to_numpy()
    for d, i in zip(date_numbers, srt):
        df.iloc[date_number_copy == d, -1] = i + 1

    # add the home_idx and away_idx columns
    team_names = np.unique(df['home'])
    team_name_indices = {}
    idx = 0
    for name in team_names:
        team_name_indices[name] = idx
        idx = idx + 1

    home_idx = np.zeros(df['home'].count()).astype(int)
    away_idx = np.zeros(df['away'].count()).astype(int)

    for i in range(len(home_idx)):
        home_idx[i] = team_name_indices[df['home'][i]]
        away_idx[i] = team_name_indices[df['away'][i]]

    df['home_idx'] = home_idx
    df['away_idx'] = away_idx

    return df, team_names

def plot_priors(mn, stdv, Home_Team_Pwr_prior_new, Home_team_name, Away_Team_Pwr_prior_new, Away_team_name):
    fig, axs = plt.subplots(2)

    axs[0].set_title(Home_team_name+' Prior')
    contours_home = axs[0].contour(mn, stdv, Home_Team_Pwr_prior_new, 5, colors='black')
    axs[0].clabel(contours_home, inline=True, fontsize=8)
    axs[0].imshow(Home_Team_Pwr_prior_new, extent=[np.min(mn), np.max(mn), np.min(stdv), np.max(stdv)], origin='lower',
               cmap='RdGy', alpha=0.5)

    axs[1].set_title(Away_team_name+' Prior')
    contours_away = axs[1].contour(mn, stdv, Away_Team_Pwr_prior_new, 5, colors='black')
    axs[1].clabel(contours_away, inline=True, fontsize=8)
    axs[1].imshow(Away_Team_Pwr_prior_new, extent=[np.min(mn), np.max(mn), np.min(stdv), np.max(stdv)], origin='lower',
               cmap='RdGy', alpha=0.5)

    fig.show()
    return 1