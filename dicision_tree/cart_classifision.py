import numpy as np
import pandas as pd

class node:
	def __init__(self, eigein, val, label):
		self.eigein = eigein
		self.label = label
		self.val =val
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
			return node("class", None, max(dic_D.items(), key=lambda x : x[1])[0])
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
		dic_D = self.hash(Data_set[ouput].values)
		T = node(min_A, min_a, max(dic_D.items(), key=lambda x : x[1])[0])
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

	def get_leaves(self, T):
		if T == None:
			return []
		if T.yes == None and T.no == None:
			return [T]
		return self.get_leaves(T.yes) + self.get_leaves(T.no)

	def prediction_err(self, T, Data_set, total_num, list_eigein, ouput):
		if T.eigein == ouput:
			dic_D = self.hash(Data_set[ouput].values)
			s = sum(dic_D.values())
			p_k = 0
			for key, val in dic_D.items():
				p_k += (val/s) * (1 - val/s)
			return (s/total_num)*p_k
		list_eigen.remove(T.eigein)
		Data_yes = pd.DataFrame(columns=list_eigen+[ouput])
		Data_no = pd.DataFrame(columns=list_eigen+[ouput])
		for i in range(Data_set.shape[0]):
			if Data_set.loc[i, T.eigein] == T.val:
				Data_yes = Data_yes.append(Data_set.loc[i, list_eigein+[ouput]])
			else:
				Data_no = Data_no.append(Data_set.loc[i, list_eigein+[ouput]])
		return self.prediction_err(T.yes, Data_yes, total_num, list_eigein, ouput) + self.prediction_err(T.no, Data_no, total_num, list_eigein, ouput)

	def get_loss(self, alpha, T, Data_set, list_eigein, ouput):
		return self.prediction_err(T, Data_set, Data_set.shape[0], list_eigein, ouput) + alpha * len(self.get_leaves(T))

	def get_node_prerr(self, t, Data_set, list_eigein, ouput):
		if t == None or t.eigein == ouput:
			return
		yes = []
		no = []
		for i in range(Data_set.shape[0]):
			if Data_set.loc[i, T.eigein] == T.val:
				yes.append(Data_set.loc[i, ouput])
			else:
				no.append(Data_set.loc[i, ouput])
		return self.Gini(yes, no)

	def tranvers_tree(self, T, Data_set, list_eigein, ouput):
		if T == None or T.eigein == ouput:
			return
		tmp_dic = {T:(self.get_node_prerr(T, Data_set, list_eigein, ouput) - self.prediction_err(T, Data_set, Data_set.shape[0], list_eigein, ouput)) / (len(self.get_leaves(T))-1)} 
		if T.yes.eigein == ouput and T.no.eigein == ouput:
			return tmp_dic
		yes_dic = self.tranvers_tree(T.yes, Data_yes, list_eigein, ouput, alpha)
		no_dic = self.tranvers_tree(T.no, Data_yes, list_eigein, ouput, alpha)
		for key, val in yes_dic.items():
			if key not in tmp_dic:
				tmp_dic[key] = val
		for key, val in no_dic.items():
			if key not in tmp_dic:
				tmp_dic[key] = val
		return tmp_dic

	def pruning(self, T, Data_set, list_eigein, ouput):#cart剪枝
		if T.eigein == ouput:
			return
		if T.yes.eigein == ouput and T.no.eigein == ouput:
			return {k:T}

		dic = self.tranvers_tree(T, Data_set, list_eigein, ouput)
		T_k, alpha_k = min(dic.items(), key=lambda x:x[1])
		T_k.eigein = ouput
		tmp_yes = T_k.yes
		tmp_no = T_k.no
		T_k.yes = None
		T_k.no = None
		del tmp_yes
		del tmp_no
		forest = {}
		forest[alpha_k] = T
		next_forest = self.pruning(T, Data_set, list_eigein, ouput)
		for key, val in next_forest.items():
			if key not in forest:
				forest[key] = val
		return forest
		











