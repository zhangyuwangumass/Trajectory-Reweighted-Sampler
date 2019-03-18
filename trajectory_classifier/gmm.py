import numpy as np
import torch


class GaussianMixture(torch.nn.Module):

	def __init__(self, n_components, n_features, n_iter=500, tol=1e-5):
		super(GaussianMixture, self).__init__()

		self.n_components, self.n_features = n_components, n_features

		#init mu
		self.mu = torch.nn.Parameter(torch.rand(self.n_components, self.n_features), requires_grad=False)
		
		#init sigma
		self.var = torch.nn.Parameter(torch.ones(self.n_components, self.n_features), requires_grad=False)

		#init pi
		self.pi = torch.nn.Parameter(torch.Tensor(self.n_components,1), requires_grad=False).fill_(1./self.n_components)

		self.n_iter = n_iter
		self.tol = tol

	def log_gaussian(self, x, mu, var):
		gaussian = torch.distributions.normal.Normal(mu,torch.sqrt(var))
		return gaussian.log_prob(x)

	def log_pk(self, X):
		log_likelihoods = self.log_gaussian(
			X[None, :, :], 
			self.mu[:, None, :], 
			self.var[:, None, :] 
		)
		return log_likelihoods.sum(-1)

	def log_marginal_likelihood(self, X):
		log_pk = torch.log(self.pi) + self.log_pk(X)
		max_log_pk = torch.max(log_pk, 0, keepdim=True)[0]
		return torch.mean(max_log_pk + torch.log(torch.sum(torch.exp(log_pk-max_log_pk), 0, keepdim=True))).numpy() # avg log marginal likelihood !important

	def fit(self, X, iprint=False):

		diff = self.tol 
		self.score = 0
		i = 0

		Xt = torch.tensor(X,dtype=torch.float32)

		while (i < self.n_iter) and (diff >= self.tol):

			pre_score = self.score

			# E-Step
			log_pk = torch.log(self.pi) + self.log_pk(Xt)
			max_log_pk = torch.max(log_pk, 0, keepdim=True)[0]
			log_r_ik = log_pk - (max_log_pk + torch.log(torch.sum(torch.exp(log_pk-max_log_pk), 0, keepdim=True)))
			r_ik = torch.exp(log_r_ik)

			# M-Step
			dem = torch.sum(r_ik, 1, keepdim=True)
			#max_log_r_ik = torch.max(log_r_ik, 1, keepdim=True)[0]
			#log_dem = max_log_r_ik + torch.log(torch.sum(torch.exp(log_r_ik - max_log_r_ik),1,keepdim=True))
			#dem = torch.exp(log_dem)

			self.pi.data = torch.div(dem, Xt.shape[0])
			self.mu.data = torch.div(torch.mm(r_ik, Xt), dem)

			for idx in range(self.n_components):

				sqr = (Xt - self.mu.data[idx])**2

				tmp = torch.mm(r_ik[idx].unsqueeze(0), sqr)

				self.var.data[idx] = torch.div(tmp, dem[idx])

			self.score = self.log_marginal_likelihood(Xt)

			diff = abs(self.score - pre_score)
			i += 1

			if iprint:
				print("epoch = {}, log_likelihood = {}".format(i, self.score))

	def predict(self, X, prob=True):
		Xt = torch.tensor(X,dtype=torch.float32)

		pk = torch.exp(self.log_pk(Xt))
		if prob:
			return (pk / torch.sum(pk)).numpy()
		else:
			_, predictions = torch.max(pk, 0)
			return torch.squeeze(predictions).type(torch.LongTensor).numpy()

	def get_model(self):
		return self.mu, self.var