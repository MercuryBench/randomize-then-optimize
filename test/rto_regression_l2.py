from __future__ import division
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
import sys 
sys.path.append('..')
from rto import *

np.random.seed(100872)

# ground truth parameter
thetatruth = np.array([0.5, 1.0, 0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

N_modes = int((len(thetatruth)-1)/2)

# weight functions to penalize high Fourier modes
#weights_cos = np.ones((N_modes,)) # no penalization
weights_cos = 1/np.arange(1, N_modes+1)
#weights_sin = np.ones((N_modes,)) # no penalization
weights_sin = 1/np.arange(1, N_modes+1)

# forward function and Jacobian
def f_fnc(theta, xs):
	N_modes = int((len(theta)-1)/2)
	temp = theta[0]
	for k in range(N_modes):
		temp += theta[k+1] * weights_cos[k]*np.cos((k+1)*xs)
	for k in range(N_modes):
		temp += theta[k+N_modes+1] * weights_sin[k]*np.sin((k+1)*xs)
	return temp

def Jf_fnc(theta, xs):
	N_modes = int((len(theta)-1)/2)
	temp = np.zeros((len(xs),2*N_modes+1))
	temp[:, 0] = np.ones((len(xs),))
	for k in range(N_modes):
		temp[:,k+1] = weights_cos[k]*np.cos((k+1)*xs)
	for k in range(N_modes):
		temp[:, k+N_modes+1] = weights_sin[k]*np.sin((k+1)*xs)
	return temp

# observation positions
xObs = np.concatenate((np.array([0, 0.2, 0.8, pi/2, 1.7, 1.8, 2.4, pi]), np.random.uniform(2, 3, (20,))), axis=0)

N = len(xObs)



# forward function for fixed observation positions
def f(theta):
	return f_fnc(theta, xObs)

def Jf(theta):
	return Jf_fnc(theta, xObs)

# observational noise standard deviation
sigma = 0.05

# generate data
y = f_fnc(thetatruth, xObs) + np.random.normal(0, sigma, (len(xObs),))

# prior standard deviation
gamma = 0.1/sqrt(2)


def cost(theta, y_aug):
	r = resf(theta, y_aug)
	return 0.5*np.dot(r.T, r)

# starting point for optimization
theta0 = np.random.normal(0, gamma, thetatruth.shape)

# RTO sampling
N_samples = 100
res = rto(f, Jf, y, sigma, gamma, theta0, mean_theta = None, N_samples=N_samples, init_method="previous")

# extract data
samples_plain = res["samples_plain"]
samples_corrected = res["samples_corrected"]
thetaMAP = res["thetaMAP"]

#plot results
xx = np.arange(0, pi, 0.01)
yy = f_fnc(thetatruth, xx)

plt.figure(1); plt.clf();plt.ion()
for n in range(17):
	plt.plot(xx, f_fnc(samples_corrected[np.random.randint(N_samples), :], xx), '0.8')

plt.plot(xx, f_fnc(thetaMAP, xx), 'k')
plt.plot(xx, f_fnc(thetatruth, xx), 'g')

plt.plot(xx, yy, 'g')
plt.plot(xObs, y, 'r.', markersize=10)
for n, pos in enumerate(xObs):
	plt.plot(np.array([pos, pos]), np.array([y[n]-2*sigma, y[n]+2*sigma]), 'r', linewidth=2)

plt.figure(2);plt.clf()
for n in range(17):
	plt.plot(samples_corrected[np.random.randint(N_samples), :], '0.8', marker=".")

plt.plot(thetaMAP.flatten(), '.k-')
plt.plot(thetatruth.flatten(), '.g-')

plt.show()


