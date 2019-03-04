from __future__ import division
import numpy as np
import scipy.optimize as opt
from math import exp, log, sqrt, pi

# python implementation for RTO
# strongly influenced by Marko Laine's Matlab code
# http://helios.fmi.fi/~lainema/rto/

# for calculation of logweights (for MH correction)
def logweight(jac, resid, Q):
	QJ = np.dot(Q.T, jac)
	try:
		L = np.linalg.cholesky(np.dot(QJ.T, QJ))
		diagonals = np.sum(np.log(np.diag(L)))
		residterm1 = np.dot(resid.T, resid)
		multresid = np.dot(Q.T, resid)
		residterm2 = np.dot(multresid.T, multresid)
		return diagonals + 0.5*(residterm1 - residterm2).flatten()
	except np.linalg.linalg.LinAlgError as err:
		# matrix is not positive definite
		return np.inf

# MH correction
def rto_accept_log(logweights):
	N = len(logweights)
	acce = np.zeros((N,), dtype="int")
	acce[0] = 1;
	ii = 1;
	ratio = 1;
	for i in range(1, N):
		if logweights[ii] - logweights[i] > log(np.random.uniform()):
			# accept
			ii = i
			ratio = ratio+1
		acce[i] = ii
	ratio = ratio/N
	return {"acce": acce, "ratio": ratio}

def rto(f_fwd, Jf_fwd, y, sigma, gamma, theta0, mean_theta=None, N_samples=1000, init_method="random"):
	# method for taking starting point for optimization
	# option:
	# "previous": take previous sample as starting point
	# "random": random starting point ---> seems to work better for multimodal distributions
	# "fixed": always take theta0
	
	if mean_theta == None:
		mean_theta = np.zeros((len(theta0),))

	# only relevant for random starting point
	if init_method == "random":
		randomization_step = 1
		randinit = np.random.normal(0, randomization_step, (len(theta0), N_samples))

	# build augmented versions (includes prior information and regularization)
	y_aug = np.concatenate((y/sigma, mean_theta/gamma), axis=0)

	def f_aug(theta):
		return np.concatenate((f_fwd(theta)/sigma, theta.T/gamma), axis=0)

	def Jf_aug(theta):
		return np.concatenate((Jf_fwd(theta)/sigma, 1/gamma*np.eye(theta.size)), axis=0)

	def resf(theta, y_aug):
		return f_aug(theta)-y_aug

	def cost(theta, y_aug):
		m = resf(theta, y_aug)
		return 0.5*np.dot(m.T, m)

	# compute starting point (MAP estimator)
	#resf_foropt = lambda theta: resf(np.reshape(theta, (1,-1)), y_aug).flatten();
	#Jf_aug_foropt = lambda theta: Jf_aug(np.reshape(theta, (1,-1)))
	res = opt.least_squares(lambda theta: resf(theta, y_aug), theta0, Jf_aug)
	thetaMAP = res.x

	Q, R = np.linalg.qr(Jf_aug(thetaMAP))

	samples = np.zeros((N_samples, theta0.size))
	logweights = np.zeros((N_samples,))
	num_bad_opts = 0;
	num_bad_QR = 0
	for k in range(N_samples):
		y_pert = y_aug + np.random.normal(0, 1, y_aug.shape)
		
		# decide for initialization point
		if init_method == "random":
			initpoint = randinit[:, k]
		elif init_method == "previous" and k >= 1:
			initpoint = samples[k-1, :]
		else: # catch-all including "fixed"
			initpoint = theta0
		res = opt.least_squares(lambda theta: np.dot(Q.T, resf(theta, y_pert)), initpoint, lambda theta: np.dot(Q.T, Jf_aug(theta)))
	
		theta = res.x
		
		if res.cost > 1e-8:
			# optimization failed
			num_bad_opts += 1
			logweights[k] = np.inf
		else:
			resid = resf(theta, y_aug)
			jac = Jf_aug(theta)
			logweights[k] = logweight(jac, resid, Q)
			if logweights[k] == np.inf:
				num_bad_QR += 1
			samples[k, :] = theta
	
	res_accept = rto_accept_log(logweights)
	acce = res_accept["acce"]
	print("accepted: " + str(res_accept["ratio"]))
	
	res = {"thetaMAP": thetaMAP, "samples_plain": samples, "samples_corrected": samples[acce, :], "logweights": logweights, "num_bad_opts": num_bad_opts, "num_bad_QR": num_bad_QR}
	return res
	
