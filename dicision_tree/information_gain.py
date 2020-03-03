import pandas as pd
import numpy as np

class information_gain:
	def get_entropy(self, X):
		num = X.shape[0]
		dic = {}
		for d in X:
			if d in dic:
				dic[d] += 1
			else:
				dic[d] = 1
		# print(dic)
		H_X = 0
		for key, val in dic.items():
			H_X -= val/num * np.log(val/num)
		return H_X	

	def get_condition_entropy(self, Y_X, X):#Y_X是一个2xn的矩阵
		num = X.shape[0]					#Y_X中第一行存的是特征Y的特征值，第二行存的是特征X的特征值
		dic = {}
		for d in X:
			if d in dic:
				dic[d] += 1
			else:
				dic[d] = 1
		condition_entrop = 0
		for key, val in dic.items():
			Y_Xi = [Y_X[0][i] for i in range(num) if Y_X[1][i] == key]
			condition_entrop += val/num * self.get_entropy(np.array(Y_Xi))
		return condition_entrop
	def get_information_gain(self, D, A):
		D_A = np.vstack((D, A))
		H_D = self.get_entropy(D) #特征D的经验熵
		H_D_A =  self.get_condition_entropy(D_A, A) #特征A对特征D的条件熵
		ig = H_D - H_D_A #信息增益
		ig_rate = ig / H_D #信息增益比
		return ig, ig_rate


# if __name__ == '__main__':
# 	ig = information_gain()
# 	print(ig.get_entropy([1, 1, 1, 0]))
