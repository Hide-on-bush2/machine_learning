from __future__ import division
import numpy as np
import pandas as pd

class Bayes_classifier:
	def __init__(self, data_set, num_class, num_eigen):
		self.data_set = data_set
		self.num_node = data_set.shape[0]
		self.num_class = num_class
		self.num_eigen = num_eigen
		self.prior_probability = np.zeros(self.num_class)
		self.conditional_probability = np.zeros((self.num_class, self.num_eigen, 3))

	def learn(self):
		for node in self.data_set:
			t_class = node[self.num_eigen]
			self.prior_probability[t_class] += 1
		self.prior_probability = self.prior_probability / self.num_node

		for c in range(self.num_class):
			for node in self.data_set:
				if node[self.num_eigen] == c:
					for j in range(self.num_eigen):
						self.conditional_probability[node[self.num_eigen], j, node[j]] += 1

			self.conditional_probability[c, :, :] = self.conditional_probability[c, :, :] / (self.prior_probability[c] * self.num_node)

	def classify(self, node):
		l = []
		for c in range(self.num_class):
			p_ck = self.prior_probability[c]
			p_con = 1
			for i in range(self.num_eigen):
				p_con *= self.conditional_probability[c, i, node[i]]
			l.append(p_ck * p_con)
		idx = np.argsort(l)
		return idx[len(idx)-1]

	def get_model(self):
		return self.prior_probability, self.conditional_probability


