import numpy as np
import pandas as pd

class Perceptron:
	def __init__(self, dimension):
		self.w = np.zeros((1, dimension))
		self.b = 0
		self.step = 0.5

	def model(self, x):
		# t_x = np.transpose(x)
		# print(t_x)
		return 1 if np.dot(self.w, x) + self.b >= 0 else -1

	def classify_fault(self, node):
		# t_x = np.transpose(node[0:2])
		return (np.dot(self.w, node[0:2]) + self.b) * node[2] <= 0

	def update(self, x, y):
		self.w = self.w + self.step * y * x
		self.b = self.b + self.step * y

	def learn(self, data_set):
		fault_nodes = []
		while True:
			for node in fault_nodes:
				self.update(node[0:2], node[2])
			fault_nodes[:] = []
			for node in data_set:
				if self.classify_fault(node):
					fault_nodes.append(node)
			if not fault_nodes:
				break;

	def get_parameter(self):
		return self.w[0], self.b


class Perceptron_antithesis:
	def __init__(self, data_set):
		self.b = 0
		self.step = 0.5
		self.data_set = data_set
		self.nums = data_set.shape[0]
		self.alpha = np.zeros(self.nums)
		self.G = np.zeros((self.nums, self.nums))
		for i in range(self.nums):
			x_i = self.data_set[i][0:2]
			for j in range(self.nums):
				x_j = self.data_set[j][0:2]
				self.G[i][j] = np.dot(x_i, x_j)
		self.x = self.data_set[:, 0:2]
		self.y = self.data_set[:, 2]


	def modle(self, x):
		l = [np.dot(self.x[i], x)*self.y[i]*self.alpha[i] for i in range(self.nums)]
		f_x = np.sum(l) + self.b
		return 1 if f_x >= 0 else -1

	def classify_fault(self, i):
		l = [self.alpha[j]*self.y[j]*self.G[j, i] for j in range(self.nums)]
		f_x = np.sum(l) + self.b
		return self.y[i] * f_x <= 0

	def update(self, i):
		self.alpha[i] = self.alpha[i] + self.step
		self.b = self.b + self.step*self.y[i]

	def learn(self):
		fault_nodes = []
		while True:
			for i in fault_nodes:
				self.update(i)
			fault_nodes[:] = []
			for i in range(self.nums):
				if self.classify_fault(i):
					fault_nodes.append(i)
			if not fault_nodes:
				break;

	def get_parameter(self):
		l_w = [self.alpha[i]*self.y[i]*self.x[i] for i in range(self.nums)]
		w = np.sum(l_w, axis=0)
		return w, self.b



















