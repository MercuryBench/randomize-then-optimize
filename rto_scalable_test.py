import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from math import exp
import scipy.stats as st

np.random.seed(100872)

# forward function and Jacobian
f_fnc = lambda x, theta: theta[0]*(1 - np.exp(-theta[1]*x))

Jf_fnc = lambda x, theta: np.stack((1 - np.exp(-theta[1]*x), theta[0] * x * np.exp(-theta[1]*x)), axis=1)

# observation positions
xObs = np.array([0.3, 0.5, 1.0, 1.8, 3.3, 5.8])

# ground truth parameter
thetatruth = np.array([1.0, 0.3]) 

# forward function for fixed observation positions
f_fwd = lambda theta: f_fnc(xObs, theta)
Jf_fwd = lambda theta: Jf_fnc(xObs, theta)

# observational noise
sigma = 0.3

# generate data
y = f_fwd(thetatruth) + np.random.normal(0, sigma, xObs.shape)

# prior standard deviation (both parameters)
gamma = 0.8

############## Stuff for didactical purposes, not necessary for sampling:
# Plot of posterior negative logdensity
xs = np.linspace(0, 6, 300)

misfit = lambda theta: 1/(2*sigma**2)*np.dot((y-f_fwd(theta)).T, y-f_fwd(theta))

# find MAP optimizer (by grid iteration)
posteriorlogdensity = lambda theta: misfit(theta) + 1/(2*gamma**2)*theta[0]**2 + 1/(2*gamma**2)*theta[1]**2

N_contours =150
theta1s = np.linspace(-3, 3, N_contours)
#theta1s = np.linspace(0.6, 1.8, N_contours)
theta2s = np.linspace(-1, 3, N_contours)
#theta2s = np.linspace(0.04, 0.2, N_contours)

T1, T2 = np.meshgrid(theta1s, theta2s)
postvals = np.zeros((N_contours, N_contours))
misvals = np.zeros((N_contours, N_contours))
for n, t1 in enumerate(theta1s):
	for m, t2 in enumerate(theta2s):
	    postvals[m, n] = posteriorlogdensity(np.array([t1, t2]))
	    misvals[m, n] = misfit(np.array([t1, t2]))

indmin = np.unravel_index(np.argmin(postvals, axis=None), misvals.shape)
thetaMAP_grid = np.array([theta1s[indmin[1]], theta2s[indmin[0]]])
######################################################################



# starting point for optimization
theta0 = np.random.normal(0, gamma, thetatruth.shape)

# RTO sampling
N_samples = 1000;
mean_theta= None
init_method = "random"
res = rto_scalable(f_fwd, Jf_fwd, y, sigma, gamma, theta0, N_samples=N_samples, init_method="random")



 # you can also try init_method = "fixed" or "previous", but will work worse

# extract data
samples_plain = res["samples_plain"]
#samples_corrected = res["samples_corrected"]
thetaMAP = res["thetaMAP"]
#logweights = res["logweights"]
#num_bad_opts = res["num_bad_opts"]
#num_bad_QR = res["num_bad_QR"]


# plot results
plt.figure(1); plt.ion()
plt.title("parameter space")
plt.contourf(theta1s, theta2s, np.log(postvals), 20, cmap=plt.get_cmap("viridis"))
plt.plot(samples_plain[:, 0], samples_plain[:, 1], '.', label="samples (uncorrected)")
#plt.plot(samples_corrected[:, 0], samples_corrected[:, 1], '.', label="samples")
plt.plot(thetatruth[0], thetatruth[1], 'go', markersize=10, label="th_true")
plt.plot(thetaMAP_grid[0], thetaMAP_grid[1], 'yo', markersize=10, label="th_MAP (grid search)")

plt.plot(thetaMAP[0], thetaMAP[1], 'mo', markersize=10, label="th_MAP (optimization)")
plt.legend(numpoints=1)

plt.figure(2); plt.clf()
inds = np.random.choice(range(samples_plain.shape[0]), 100, replace=False)
plt.plot(xs, f_fnc(xs, np.reshape(samples_plain[0], thetatruth.shape)), '0.9', linewidth=1, label="samples (uncorrected)")
for ind in inds:	
	plt.plot(xs, f_fnc(xs, np.reshape(samples_plain[ind], thetatruth.shape)), '0.9', linewidth=1)

plt.plot(xObs, y, 'rx', label="observation")
plt.plot(xs, f_fnc(xs, thetatruth), 'g', linewidth=3, label="th_true")
plt.plot(xs, f_fnc(xs, thetaMAP_grid), 'y', linewidth=3, label="th_MAP (grid search)")
plt.plot(xs, f_fnc(xs, thetaMAP), 'm', linewidth=3, label="th_MAP (optimization)")
plt.legend(numpoints=1)
plt.show()

