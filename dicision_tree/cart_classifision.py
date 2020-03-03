import numpy as np
import pandas as pd

class node:
	def __init__(self, eigein, label):
		self.eigein = eigein
		self.label = label
		self.yes = None
		self.no = None

class CartClassifision:
	def hash(self, l):#统计集合中类别的个数
		dic = {}
		for item in l:
			if item in dic:
				dic[item] += 1
			else:
				dic[item] = 1
		return dic

	def Gini(self, D1, D2):#计算基尼指数（CART算法每个节点至多有两个子节点，属于二分类问题）
		dic_1 = self.hash(D1)
		dic_2 = self.hash(D2)
		sum_1 = sum(dic_1.values())
		sum_2 = sum(dic_2.values())
		gini_1 = 0
		for key, val in dic_1.items():
			gini_1 += val/sum_1 * (1 - val/sum_1)
		gini_2 = 0
		for key, val in dic_2.items():
			gini_2 += val/sum_2 * (1 - val/sum_2)
		return (sum_1/(sum_1 + sum_2))*gini_1 + (sum_2/(sum_1 + sum_2))*gini_2


	def cart_classify(self, Data_set, list_eigen, ouput):	
		num = Data_set.shape[0]
		if len(list_eigen) <= 0:
			dic_D = self.hash(Data_set[ouput].values)
			return node("class", max(dic_D.items(), key=lambda x : x[1])[0])
		dic_A_a = {}
		for eigein in list_eigen:
			dic_eigein = self.hash(Data_set[eigein].values)
			for a, val in dic_eigein.items():
				D_yes = []
				D_no = []
				for i in range(num):
					if Data_set.loc[i, eigein] == a:
						D_yes.append(Data_set.loc[i, ouput])
					else:
						D_no.append(Data_set.loc[i, ouput])
				dic_A_a[(eigein, a)] = self.Gini(D_yes, D_no)
		min_A, min_a = min(dic_A_a.items(), key=lambda x : x[1])[0]
		T = node(min_A, None)
		list_eigen.remove(min_A)
		Data_yes = pd.DataFrame(columns=list_eigen+[ouput])
		Data_no = pd.DataFrame(columns=list_eigen+[ouput])
		for i in range(num):
			if Data_set.loc[i, min_A] == min_a:
				Data_yes = Data_yes.append(Data_set.loc[i, list_eigen+[ouput]], ignore_index=True)
			else:
				Data_no = Data_no.append(Data_set.loc[i, list_eigen+[ouput]], ignore_index=True)
		T.yes = self.cart_classify(Data_yes, list_eigen, ouput)
		T.no = self.cart_classify(Data_no, list_eigen, ouput)
		return T
		
	def print_tree(self, T):
		if T == None:
			return
		print("eigein:", T.eigein)
		print("label:", T.label)
		if T.yes != None:
			print("yes->", T.yes.eigein)
		if T.no != None:
			print("no->", T.no.eigein)
		print()
		self.print_tree(T.yes)
		self.print_tree(T.no)











