import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import NFL_Functions as NFL

#### Initialize Variables
# Parameter Settings
Std = 2 # stdev of PwrX
StdPrior = 2 #Initial Stdev of prior
dx = 0.1
x = np.arange(-30,30,dx)

# Make up Power Distributions for two teams
PwrA = [7,2]
PwrA_prior = norm.pdf(x, loc=PwrA[0], scale=StdPrior)
PwrB = [-5,2]
PwrB_prior = norm.pdf(x, loc=PwrB[0], scale=StdPrior)




#### Predict Step
# Compute the predicted score distribution
SDiff_predicted = NFL.predict_outcome(PwrA, PwrB)

# assign a made up game result
SDiff = 4

#### Estimate Step - use Bayes Thm and the game outcome to compute new power distributions and priors
PwrA_new, PwrA_prior_new, PwrB_new, PwrB_prior_new = NFL.update_power_distributions(
    PwrA, PwrA_prior, PwrB, PwrB_prior, SDiff, dx, x)

# show the plots
NFL.plot_update_resutls(PwrA, PwrA_new, PwrB, PwrB_new, SDiff, SDiff_predicted, x)


