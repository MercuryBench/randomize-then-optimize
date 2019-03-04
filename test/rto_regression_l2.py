from __future__ import division
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
from rto_util import *

xs_obs = np.reshape(np.array([0, 0.2, 0.8, pi/2, 1.7, 1.8, 2.4, pi]), (-1,1))
N = len(xs_obs)

# observational noise standard deviation
sigma = 0.5

# prior standard deviation
gamma = 3


thetaTruth = np.array([[0.5, 1.0, 0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]])

plt.ion()
plt.show()


N_modes = int(((thetaTruth.size)-1)/2)

def f_fnc(theta, xs):
	temp = theta[0,0]
	for k in range(N_modes):
		temp += theta[0,k+1] * np.cos((k+1)*xs)
	for k in range(N_modes):
		temp += theta[0,k+N_modes+1] * np.sin((k+1)*xs)
	return temp

def Jf_fnc(theta, xs):
	temp = np.zeros((len(xs),2*N_modes+1))
	temp[:, 0] = np.ones((len(xs),))
	for k in range(N_modes):
		temp[:, k+1] = np.cos((k+1)*xs)[:,0]
	for k in range(N_modes):
		temp[:, k+N_modes+1] = np.sin((k+1)*xs)[:,0]
	return temp

def f(theta):
	return f_fnc(theta, xs_obs)

def Jf(theta):
	return Jf_fnc(theta, xs_obs)




def cost(theta, y_aug):
	r = resf(theta, y_aug)
	return 0.5*np.dot(r.T, r)

y = f_fnc(thetaTruth, xs_obs) + np.random.normal(0, sigma, (len(xs_obs),1))
#y_aug = np.concatenate((y/sigma, np.zeros((thetaTruth.size,1))), axis=0)

N_samples = 1000
res = rto_samples(f, Jf, y, sigma, gamma, thetaTruth, N_samples)

samples_plain = res["samples_plain"]
samples_corrected = res["samples_corrected"]
thetaMAP = res["thetaMAP"]

xx = np.arange(0, pi, 0.01)
yy = f_fnc(thetaTruth, xx)
plt.figure(1); plt.clf();plt.ion()
plt.plot(xx, yy, 'g')
plt.plot(xx, f_fnc(thetaMAP, xx), 'k')
plt.plot(xs_obs, y, 'r.', markersize=5)
plt.show()


# RTO Sampling



plt.figure(2);plt.clf()
for n in range(17):
	plt.plot(samples_corrected[np.random.randint(N_samples), :], '0.8', marker=".")

plt.plot(thetaMAP.flatten(), '.k-')
plt.plot(thetaTruth.flatten(), '.g-')


plt.figure(1)
for n in range(17):
	plt.plot(xx, f_fnc(np.reshape(samples_corrected[np.random.randint(N_samples), :], (1, -1)), xx), '0.8')

plt.plot(xx, f_fnc(thetaMAP, xx), 'k')
plt.plot(xx, f_fnc(thetaTruth, xx), 'g')

plt.plot(xx, yy, 'g')
plt.plot(xs_obs, y, 'r.', markersize=10)
for n, pos in enumerate(xs_obs):
	plt.plot(np.array([pos, pos]), np.array([y[n]-2*sigma, y[n]+2*sigma]), 'r', linewidth=2)



