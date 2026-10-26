###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn

from . import utils

#####################################################################################################

class ODEFunc(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, time_dependent=True, device=None):
		"""Initialize the ODE function with input and latent state dimensions."""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.time_dependent = time_dependent

		utils.init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""Return the gradient dy/dt for state y at time t_local."""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		if not self.time_dependent:
			return self.gradient_net(y)

		t_feature = self._time_feature(t_local, y)
		return self.gradient_net(torch.cat((y, t_feature), dim=-1))

	def _time_feature(self, t_local, y):
		if torch.is_tensor(t_local):
			t_feature = t_local.to(device=y.device, dtype=y.dtype)
		else:
			t_feature = torch.tensor(t_local, device=y.device, dtype=y.dtype)

		if t_feature.shape[-1:] != (1,):
			t_feature = t_feature.unsqueeze(-1)
		while t_feature.ndim < y.ndim:
			t_feature = t_feature.unsqueeze(0)
		return t_feature.expand(*y.shape[:-1], 1)

	def sample_next_point_from_prior(self, t_local, y):
		"""Return the prior ODE gradient for state y at time t_local."""
		return self.get_ode_gradient_nn(t_local, y)

#####################################################################################################

class ODEFunc_w_Poisson(ODEFunc):
	def __init__(self, input_dim, latent_dim, ode_func_net,
		lambda_net, device = torch.device("cpu")):
		"""Initialize the ODE function with input and latent state dimensions."""
		super(ODEFunc_w_Poisson, self).__init__(
			input_dim,
			latent_dim,
			ode_func_net,
			time_dependent=True,
			device=device,
		)

		self.latent_ode = ODEFunc(input_dim = input_dim,
			latent_dim = latent_dim,
			ode_func_net = ode_func_net,
			time_dependent=True,
			device = device)

		self.latent_dim = latent_dim
		self.lambda_net = lambda_net
		# The computation of poisson likelihood can become numerically unstable. 
		#The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
		#Exponent of lambda can also take large values
		# So we divide lambda by the constant and then multiply the integral of lambda by the constant
		self.const_for_lambda = torch.Tensor([100.]).to(device)

	def extract_poisson_rate(self, augmented, final_result = True):
		y, log_lambdas, int_lambda = None, None, None

		assert(augmented.size(-1) == self.latent_dim + self.input_dim)		
		latent_lam_dim = self.latent_dim // 2

		if len(augmented.size()) == 3:
			int_lambda  = augmented[:,:,-self.input_dim:] 
			y_latent_lam = augmented[:,:,:-self.input_dim]

			log_lambdas  = self.lambda_net(y_latent_lam[:,:,-latent_lam_dim:])
			y = y_latent_lam[:,:,:-latent_lam_dim]

		elif len(augmented.size()) == 4:
			int_lambda  = augmented[:,:,:,-self.input_dim:]
			y_latent_lam = augmented[:,:,:,:-self.input_dim]

			log_lambdas  = self.lambda_net(y_latent_lam[:,:,:,-latent_lam_dim:])
			y = y_latent_lam[:,:,:,:-latent_lam_dim]

		# Multiply the intergral over lambda by a constant 
		# only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
		if final_result:
			int_lambda = int_lambda * self.const_for_lambda
			
		# Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
		assert(y.size(-1) == latent_lam_dim)

		return y, log_lambdas, int_lambda, y_latent_lam


	def get_ode_gradient_nn(self, t_local, augmented):
		y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(augmented, final_result = False)
		dydt_dldt = self.latent_ode(t_local, y_latent_lam)

		log_lam = log_lam - torch.log(self.const_for_lambda)
		return torch.cat((dydt_dldt, torch.exp(log_lam)),-1)




