from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
from math import exp, copysign, log, sqrt, pi
import sys 
sys.path.append('..')

from rto_l1 import *

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

# Laplace prior scale gamma (std = sqrt(2)*gamma)
gamma = 0.1
lam = 1/gamma


def cost(theta, y_aug):
	r = resf(theta, y_aug)
	return 0.5*np.dot(r.T, r)

# starting point for optimization
u0 = np.random.normal(0, gamma, thetatruth.shape)

# RTO sampling
N_samples = 100

lambdas = lam*np.ones((2*N_modes+1,))

res = rto_l1(f, Jf, y, sigma, lambdas, u0, N_samples)
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


"""np.random.seed(1992)

xs_obs = np.concatenate((np.array([0, 0.2, 0.8, pi/2, 1.7, 1.8, 2.4, pi]), np.random.uniform(4, 2*pi, (30,))), axis=0)
N = len(xs_obs)
sigma = 0.2

thetaTruth = np.array([0.5, 1.0, 0, 0.1, 0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0, 0])
N_modes = int((len(thetaTruth)-1)/2)

coeffs_cos = 1/np.arange(1, N_modes+1)#np.ones((N_modes,))
coeffs_sin = 1/np.arange(1, N_modes+1)#np.ones((N_modes,))

def f_fnc(theta, xs):
	temp = theta[0]
	N_modes = int((len(theta)-1)/2)
	for k in range(N_modes):
		temp += theta[k+1] * coeffs_cos[k]*np.cos((k+1)*xs)
	for k in range(N_modes):
		temp += theta[k+N_modes+1] * coeffs_sin[k]*np.sin((k+1)*xs)
	return temp

def Jf_fnc(theta, xs):
	temp = np.zeros((len(xs),2*N_modes+1))
	temp[:, 0] = np.ones((len(xs),))
	for k in range(N_modes):
		temp[:, k+1] = coeffs_cos[k]*np.cos((k+1)*xs)
	for k in range(N_modes):
		temp[:, k+N_modes+1] = coeffs_sin[k]*np.sin((k+1)*xs)
	return temp
	
# variants with fixed x in observation points
f = lambda theta: f_fnc(theta, xs_obs)
Jf = lambda theta: Jf_fnc(theta, xs_obs)

xx = np.arange(0, 2*pi, 0.01)
yy = f_fnc(thetaTruth, xx)

y = f_fnc(thetaTruth, xs_obs) + np.random.normal(0, sigma, (len(xs_obs),))


lam = 3

def norm1(theta, lam_val):
	return lam_val*np.sum(np.abs(theta))
def FncL1(theta, y, lam_val):
	return Misfit(theta, y) + norm1(theta, lam_val)

N_iter = 300
tau = 0.002
val = np.zeros((N_iter,))

thetaOpt = np.zeros((2*N_modes+1,))


# find MAP estimator
misfit = lambda theta: f(theta)-y

def Phi_fnc(theta):
	m = misfit(theta)
	return 1/(2*sigma**2)*np.dot(m.T, m) 

def DPhi_fnc(theta):
	return np.dot(Jf(theta).T, misfit(theta))/sigma**2

I_fnc = lambda theta: Phi_fnc(theta) + norm1(theta, lam)

res = FISTA(thetaOpt, I_fnc, Phi_fnc, DPhi_fnc, 2*sigma**2*lam, alpha0=10, eta=0.5, N_iter=500, c=1.0, showDetails=True)

thetaOpt = np.copy(res["sol"])



plt.figure(2)
plt.title("FISTA")
plt.plot(res["Is"])

lambdas = lam*np.ones((2*N_modes+1,))
u0 = np.zeros((2*N_modes+1,))
N_samples = 250
res_rto = rto_l1(f, Jf, y, sigma, lambdas, u0, N_samples)

thetaMAP, samples = res_rto["thetaMAP"], res_rto["samples_corrected"]

print("thetaTruth: I = " + str(I_fnc(thetaTruth)) + " = " + str(Phi_fnc(thetaTruth)) + " (misfit) + " + str(norm1(thetaTruth, lam)) + " (norm)")
print("thetaMAP(sampling): I = " + str(I_fnc(thetaMAP)) + " = " + str(Phi_fnc(thetaMAP)) + " (misfit) + " + str(norm1(thetaMAP, lam)) + " (norm)")
print("thetaOpt(FISTA): I = " + str(I_fnc(thetaOpt)) + " = " + str(Phi_fnc(thetaOpt)) + " (misfit) + " + str(norm1(thetaOpt, lam)) + " (norm)")











plt.figure(3);
for n in range(17):
	plt.plot(samples[np.random.randint(N_samples), :], '0.8', marker=".")
plt.plot(thetaMAP, '.k-', label="th_MAP (from sampling)")
plt.plot(thetaTruth, '.g-', label="th_true")
plt.plot(thetaOpt, '.b-', label="th_OPT (from FISTA)")
plt.legend()



plt.figure(1);plt.ion()
plt.plot(xs_obs, y, 'r.', markersize=10, label="obs")
plt.plot(xx, f_fnc(thetaTruth, xx), 'g', label="th_true")
for n in range(17):
	plt.plot(xx, f_fnc(samples[np.random.randint(N_samples), :], xx), '0.8')

plt.plot(xx, f_fnc(thetaMAP, xx), 'k', label="th_MAP (from sampling)")
	

plt.plot(xx, yy, 'g')
plt.plot(xs_obs, y, 'r.', markersize=10)

plt.plot(xx, f_fnc(thetaOpt, xx), 'b', label="th_OPT (from FISTA)")
plt.legend()
plt.show()

"""
