import numpy as np
import scipy.optimize as opt
from math import exp, log, sqrt, pi
import scipy
import time
# python implementation for RTO
# see Wang, Zheng, et al. "Scalable optimization-based sampling on function space." arXiv preprint arXiv:1903.00870 (2019).


def rto_accept(weights):
	N = len(weights)
	acce = np.zeros((N,), dtype="int")
	acce[0] = 1;
	ii = 1;
	ratio = 1;
	thresh = np.random.uniform(0, 1, (N,))
	for i in range(1, N):
		if weights[i]/weights[ii] > thresh[i]:
			# accept
			ii = i
			ratio = ratio+1
		acce[i] = ii
	ratio = ratio/N
	return {"acce": acce, "ratio": ratio}

def rto_scalable(f_fwd, Jf_fwd, y, sigma, Gamma, theta0, mean_theta=None, N_samples=1000, init_method="random", opt_with_grad=True): 
	# method for taking starting point for optimization
	# option:
	# "previous": take previous sample as starting point
	# "random": random starting point ---> seems to work better for multimodal distributions
	# "fixed": always take theta0
	
	# opt_with_grad: Boolean on whether the optimization by scipy is supposed to employ gradient info as well.
	Spr = scipy.linalg.sqrtm(Gamma)

	# transform variables: v = (theta-mean_theta)/gamma, G(v) = (f_fwd(gamma*v-mean_theta)-y)/sigma, e = eps/sigma
	if mean_theta is None:
		mean_theta = np.zeros((len(theta0),))

	def G(v):
		return (f_fwd(Spr @ v+mean_theta)-y)/sigma

	def DG(v):
		return (Jf_fwd(Spr @ v+mean_theta)) @ Spr/sigma

	def H(v):
		return np.concatenate((v, G(v)), axis=0)

	def DH(v):
		return np.concatenate((np.eye(len(v)), DG(v)), axis=0)

	if opt_with_grad:
		res = opt.least_squares(H, theta0, DH)
	else:
		res = opt.least_squares(H, theta0)
	
	vMAP = res.x


	

	# compute SVD
	Psi, Lambda, PhiT = np.linalg.svd(DG(vMAP), full_matrices=False)

	r = Lambda.shape[0]

	# DH(v_MAP)
	DHvMAP = DH(vMAP)

	# preconditioner
	Q = np.dot(DHvMAP, np.linalg.inv(scipy.linalg.sqrtm(np.dot(DHvMAP.T, DHvMAP))))


	# utility for weight function	
	def costterm1(v):
		t = H(v)
		return 0.5*np.dot(t, t)
	
	def costterm2(v):
		t = Q.T @ H(v)
		return 0.5*np.dot(t, t)

	def weightfunction(v):
		t1 = exp(-costterm1(v)+costterm2(v))
		t2 = np.prod(1/np.sqrt(1 + Lambda))
		t3 = np.linalg.det(np.eye(r) + Lambda*(Psi.T @ DG(v) @ Phi))
		return np.abs(t1*t2*t3)

	samples = np.zeros((N_samples, theta0.size))
	weights = np.zeros((N_samples,))
	xi = np.random.normal(0,1,(N_samples,len(vMAP)))
	Phi = PhiT.T	
	s1 = time.time()

	# only relevant for random starting point
	if init_method == "random":
		randomization_step = 1
		randinit = np.random.normal(0, randomization_step, (r, N_samples))
	# start sampling
	for k in range(N_samples):
		# [consult paper for the following formulae]
		proj = xi[k, :] - np.dot(Phi, np.dot(PhiT, xi[k, :]))
		# decide for initialization point
		if init_method == "random":
			#print("random")
			initpoint = randinit[:, k]
		elif init_method == "previous" and k >= 1:
			#print("previous")
			initpoint = samples[k-1, 0:r]
		else: # catch-all including "fixed"
			#print("fixed")
			initpoint = theta0[0:r]

		# now solve optimization problem min \|I(v)\|^2
		temp1 = 1/np.sqrt(1 + Lambda)
		def I(v):		
			return temp1*v + Lambda*temp1* ( Psi.T @ G(proj + Phi @ v)) - PhiT @ xi[k, :] 
			#return np.dot(temp1, v) + np.dot(Lambda, np.dot(temp1, np.dot(Psi.T,G(proj + np.dot(Phi, v))))) - np.dot(PhiT, xi[k, :])
		def DI(v):
			return np.diag(temp1) + np.diag(Lambda*temp1) @ ( Psi.T @ DG(proj + Phi @ v)) @ Phi
	#return temp1 + np.dot(np.dot(Lambda, np.dot(temp1, np.dot(Psi.T,DG(proj + np.dot(Phi, v) ) ) ) ), Phi)
		s3 = time.time()
		if opt_with_grad:		
			res = opt.least_squares(I, initpoint, jac=DI)
		else:	
			res = opt.least_squares(I, initpoint)
		s4 = time.time()
		#print(s4-s3)
		#print(res)
		
		v = Phi @ res.x + proj
		s5 = time.time()
		weights[k] = weightfunction(v)
		s6 = time.time()
		#print("weights: " + str(s6-s5))
		samples[k, :] = Spr @ v + mean_theta
	
	s2 = time.time()
	#print(s2-s1)
	res_accept = rto_accept(weights)
	acce = res_accept["acce"]
	print("accepted: " + str(res_accept["ratio"]))
	
	thetaMAP = Spr @ vMAP + mean_theta
	
	res = {"thetaMAP": thetaMAP, "samples_plain": samples, "samples_corrected": samples[acce, :], "weights": weights}
	return res

if __name__ == "__main__":
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
	thetatruth = np.array([3.0, 0.3]) 

	# forward function for fixed observation positions
	f_fwd = lambda theta: f_fnc(xObs, theta)
	Jf_fwd = lambda theta: Jf_fnc(xObs, theta)

	# observational noise
	sigma = 0.7

	# generate data
	y = f_fwd(thetatruth) + np.random.normal(0, sigma, xObs.shape)

	# prior standard deviation (both parameters)
	Gamma = np.diag([0.8**2, 0.8**2]) # was: 0.8, 0.8
	mean_theta = np.array([0.5, 0.35])

	############## Stuff for didactical purposes, not necessary for sampling:
	# Plot of posterior negative logdensity
	xs = np.linspace(0, 6, 300)

	misfit = lambda theta: 1/(2*sigma**2)*np.dot((y-f_fwd(theta)).T, y-f_fwd(theta))

	# find MAP optimizer (by grid iteration)
	posteriorlogdensity = lambda theta: misfit(theta) + 0.5*(theta-mean_theta) @ (np.linalg.inv(Gamma) @ (theta-mean_theta))# 1/(2*gamma**2)*theta[0]**2 + 1/(2*gamma**2)*theta[1]**2

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
	theta0 = np.random.multivariate_normal(mean_theta, Gamma)
	#theta0 = thetaMAP_grid
	# RTO sampling
	N_samples = 300;
	#mean_theta= None
	init_method = "random"
		
	res = rto_scalable(f_fwd, Jf_fwd, y, sigma, Gamma, theta0, mean_theta=mean_theta, N_samples=N_samples, init_method=init_method, opt_with_grad=True)

	
	
	 # you can also try init_method = "fixed" or "previous", but will work worse

	# extract data
	samples_plain = res["samples_plain"]
	samples_corrected = res["samples_corrected"]
	thetaMAP = res["thetaMAP"]
	#logweights = res["logweights"]
	#num_bad_opts = res["num_bad_opts"]
	#num_bad_QR = res["num_bad_QR"]

	# plot results
	plt.figure(); plt.ion()
	plt.title("parameter space")
	plt.contourf(theta1s, theta2s, np.log(postvals), 20, cmap=plt.get_cmap("viridis"))
	plt.plot(samples_plain[:, 0], samples_plain[:, 1], '.', label="samples (uncorrected)")
	plt.plot(samples_corrected[:, 0], samples_corrected[:, 1], '.', label="samples")
	plt.plot(thetatruth[0], thetatruth[1], 'go', markersize=10, label="th_true")
	plt.plot(thetaMAP_grid[0], thetaMAP_grid[1], 'yo', markersize=10, label="th_MAP (grid search)")

	plt.plot(thetaMAP[0], thetaMAP[1], 'mo', markersize=10, label="th_MAP (optimization)")
	plt.legend(numpoints=1)

	plt.figure(2); plt.clf()
	inds = np.random.choice(range(samples_corrected.shape[0]), 100, replace=False)
	plt.plot(xs, f_fnc(xs, np.reshape(samples_corrected[0], thetatruth.shape)), '0.9', linewidth=1, label="samples (corrected)")
	for ind in inds:	
		plt.plot(xs, f_fnc(xs, np.reshape(samples_corrected[ind], thetatruth.shape)), '0.9', linewidth=1)

	plt.plot(xObs, y, 'rx', label="observation")
	plt.plot(xs, f_fnc(xs, thetatruth), 'g', linewidth=3, label="th_true")
	plt.plot(xs, f_fnc(xs, thetaMAP_grid), 'y', linewidth=3, label="th_MAP (grid search)")
	plt.plot(xs, f_fnc(xs, thetaMAP), 'm', linewidth=3, label="th_MAP (optimization)")
	plt.legend(numpoints=1, loc=4)
	plt.show()


	
