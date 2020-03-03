import operator
import numpy as np
import pandas as pd

class node:
		def __init__(self, sp_value, sp_dot, c_m):
			self.sp_value = sp_value
			self.sp_dot = sp_dot
			self.c_m = c_m
			self.yes = None
			self.no = None

class CartRegression:
	def square_loss(self, arr):
		aver = np.average(arr)
		loss = 0
		for item in arr:
			loss += (aver - item) ** 2
		return loss

	def hash(self, l):#统计集合中类别的个数
		dic = {}
		for item in l:
			if item in dic:
				dic[item] += 1
			else:
				dic[item] = 1
		return dic

	def cart_regres(self, Data_set, list_eigen, ouput, c_m):
		if len(list_eigen) <= 0:
			return node(None, None, c_m)
		num = Data_set.shape[0]
		dic_eigen = {}
		for c in list_eigen:#遍历切分变量j
			t_dic = self.hash(Data_set[c].values)
			sq_val = {}
			for key in t_dic:#确定最优的切分点
				yes = []
				no = []
				for i in range(num):
					if Data_set.loc[i, c] <= key:
						no.append(Data_set.loc[i, ouput])
					else:
						yes.append(Data_set.loc[i, ouput])
				sq_val[key] = self.square_loss(yes) + self.square_loss(no)
			dic_eigen[c] = min(sq_val.items(), key=lambda x : x[1]) #选择最小的作为切分变量c的损失值
		min_js = min(dic_eigen.items(), key=operator.itemgetter(1,1))
		min_j = min_js[0]
		min_s = min_js[1][0]#得到最小的切分变量和切分点

		T = node(min_j, min_s, c_m) #创建当前节点

		list_eigen.remove(min_j)
		df_no = pd.DataFrame(columns=list_eigen)
		df_yes = pd.DataFrame(columns=list_eigen)
		for i in range(num):#划分特征空间
			if Data_set.loc[i, min_j] <= min_s:
				df_no = df_no.append(Data_set.loc[i, list_eigen + [ouput]], ignore_index=True)
			else:
				df_yes = df_yes.append(Data_set.loc[i, list_eigen + [ouput]], ignore_index=True)
		c_m_no = np.average(df_no[ouput].values)#得到每个空间的c_m值
		c_m_yes = np.average(df_yes[ouput].values)
		T.no = self.cart_regres(df_no, list_eigen, ouput, c_m_no)#递归对下面的空间进行划分
		T.yes = self.cart_regres(df_yes, list_eigen, ouput, c_m_yes)

		return T

	def print_tree(self, T):
		if T == None:
			return
		print("spilit value:", T.sp_value)
		print("spilit dot:", T.sp_dot)
		print("c_m value:", T.c_m)
		if T.no != None:
			print("no->", T.no.sp_value)
		if T.yes != None:
			print("yes->", T.yes.sp_value)
		self.print_tree(T.no)
		self.print_tree(T.yes)
		print()

	def get_regression_value(self, T, x):
		if T.sp_value == None:
			return T.c_m
		if x[T.sp_value].values[0] <= T.sp_dot:
			return T.c_m + self.get_regression_value(T.no, x)
		else:
			return T.c_m + self.get_regression_value(T.yes, x)

