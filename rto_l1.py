from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
from math import exp, copysign, log, sqrt
from scipy.special import erfc
from rto import *


np.random.seed(100872)


def rto_l1(f, Jf, y, sigma, lambdas, u0, N_samples = 1000, init_method="random"):
	# transform variable
	def g(u):
		return -1/lambdas*np.sign(u)*np.log(erfc(np.abs(u)/sqrt(2)))

	def Jg(u):
		vec_dgs = []
		for n, lam in enumerate(lambdas):
			vec_dgs.append(st.norm._pdf(u[n])/(lam*st.norm._cdf(-abs(u[n]))))
		return np.diag(vec_dgs)


	# new forward problem with transformed variable
	def f_tilde(u):
		return f(g(u))/sigma

	def Jf_tilde(u):
		return np.dot(Jf(g(u)), Jg(u))/sigma

	y_tilde = 1/sigma*y

	
	res = rto(f_tilde, Jf_tilde, y_tilde, 1, 1, u0, N_samples=N_samples, init_method=init_method)
	
	samples_plain = res["samples_plain"]
	samples_corrected = res["samples_corrected"]
	uMAP = res["thetaMAP"] # note that variables in rto_util are also called theta, hence the slight name confusion
	logweights = res["logweights"]
	num_bad_opts = res["num_bad_opts"]
	num_bad_QR = res["num_bad_QR"]
	# transform into theta space:
	samples_tr_corrected = np.zeros_like(samples_corrected)
	for k in range(samples_corrected.shape[0]):
		samples_tr_corrected[k, :] = g(samples_corrected[k, :])
	samples_tr_plain = np.zeros_like(samples_plain)
	for k in range(samples_plain.shape[0]):
		samples_tr_plain[k, :] = g(samples_plain[k, :])
	thetaMAP = g(uMAP)
	
	res_new = {"thetaMAP": thetaMAP, "samples_plain": samples_tr_plain, "samples_corrected": samples_tr_corrected, "logweights": logweights, "num_bad_opts": num_bad_opts, "num_bad_QR": num_bad_QR}

	return res_new

# Example: MOD model

if __name__ == "__main__":
	def f_fnc(x, theta):
		return theta[0]*(1 - np.exp(-theta[1]*x))

	def Jf_fnc(x, theta):
		return np.stack((1 - np.exp(-theta[1]*x), theta[0] * x * np.exp(-theta[1]*x)), axis=1)


	def f(theta):
		return f_fnc(xObs, theta)

	def Jf(theta):
		return Jf_fnc(xObs, theta)
	
	# ref values
	sigma = 0.3 # 0.02
	lambdas = np.array([1.0, 0.5])
	thetatruth = np.array([1.0, 0.3]) # u= 0.9, 0.175
	xObs = np.array([0.3, 0.5, 1.0, 1.8, 3.3, 5.8])


	"""sigma = 0.5 # 0.02
	lambdas = np.array([1, 1])
	thetatruth = np.array([1.0, 0.3]) # u=0.1195, 0.0731
	xObs = np.array([0.3, 0.5, 1.0, 1.8, 3.3])"""

	xVec = np.linspace(0, 6, 100)
	
	# make observation
	y = f(thetatruth) + np.random.normal(0, sigma, (6,))
	
	u0 = np.array([0.001, 0.001])
	N_samples = 1000
	res = rto_l1(f, Jf, y, sigma, lambdas, u0, N_samples = N_samples, init_method="random")
	
	t1vec = np.linspace(-2.0, 3.0, 200)
	t2vec = np.linspace(-0.5, 5.0, 200)
	post = np.zeros((200,200))
	for mm in range(200):
		for nn in range(200):
			th = np.array([t1vec[mm], t2vec[nn]])
			misfit = f(th) - y
			post[mm, nn] = 1/(2.0*sigma**2)*np.dot(misfit.T, misfit) + np.dot(np.abs(th), lambdas)
	plt.figure(2); plt.ion()
	plt.subplot(311)
	samples = res["samples_corrected"]
	samples_plain = res["samples_plain"]
	thetaMAP = res["thetaMAP"]
	plt.contourf(t1vec, t2vec, np.exp(-post).T, cmap=plt.get_cmap("Blues"))
	plt.plot(samples[:, 0], samples[:, 1], '.', markersize=2)
	#plt.plot(0.1195, 0.0731, 'go', markersize=10)
	plt.plot(thetaMAP[0], thetaMAP[1], 'ro', markersize=10)
	plt.plot(thetatruth[0], thetatruth[1], 'go', markersize=10)
	plt.subplot(312)
	plt.plot(samples[:, 0])
	plt.subplot(313)
	plt.plot(samples[:, 1])
	plt.show()
	
	plt.figure(1)
	inds = np.random.choice(range(samples.shape[0]), 70, replace=False)
	for ind in inds:
		plt.plot(xVec, f_fnc(xVec, samples[ind, :]), '0.8')
	plt.plot(xVec, f_fnc(xVec, thetatruth), 'g', linewidth=3)
	plt.plot(xVec, f_fnc(xVec, thetaMAP), 'r', linewidth=3)
	plt.plot(xObs, y, 'rx')
	
	plt.figure(); plt.plot(res["logweights"]); plt.title("logweights")
	"""# plotting samples with KDE:
	import scipy.stats as st
	t1s, t2s = np.mgrid[0:3:0.005, 0.1:5:0.005]
	positions = np.vstack([t1s.ravel(), t2s.ravel()])
	kernel = st.gaussian_kde(samples_tr.T)
	ff = np.reshape(kernel(positions), t1s.shape)
	plt.figure()
	plt.subplot(211)
	plt.title("posterior")
	plt.contourf(t1vec, t2vec, np.exp(-post).T, cmap=plt.get_cmap("Blues"))
	plt.subplot(212)
	plt.title("kernel density estimate of samples")
	plt.contourf(t1s, t2s, ff, cmap="Blues")"""


	
	
