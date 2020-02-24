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


