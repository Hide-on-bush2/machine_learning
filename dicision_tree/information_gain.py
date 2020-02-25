import pandas as pd
import numpy as np

class information_gain:
	def get_entropy(self, data):
		num = len(data)
		dic = {}
		for d in data:
			if d in dic:
				dic[d] += 1
			else:
				dic[d] = 1
		print(dic)
		H_X = 0
		for key, val in dic.items():
			H_X -= val/num * np.log(val/num)
		return H_X	




if __name__ == '__main__':
	ig = information_gain()
	print(ig.get_entropy([1, 1, 1, 0]))