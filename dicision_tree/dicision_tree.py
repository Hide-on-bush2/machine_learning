import numpy as np
import pandas as pd
import operator
from information_gain import information_gain


class node:
	def __init__(self, engein, num_engein, label):
		self.engein = engein #用于决策的特征
		self.children = {}	
		self.label = label	#当前节点的标签（非叶子结点的标签好像没多大用）
		for i in range(num_engein):
			self.children[i] = None

class Dicision_tree:
	def ID3(self, Data_set, e, label, list_A):
		D = Data_set.loc[:, ['class']] #类集
		A = Data_set.loc[:, list_A]	#特征集
		num = D.shape[0]
		class_D = self.hash(D.values.reshape((1, num))[0])
		if len(class_D) <= 1: #如果类别只有一种，返回单节点树，标签为这个类别
			return node("class", 2, list(class_D.keys())[0])

		if len(list_A) <= 0: #如果特征都分完了，返回单节点树，标签为类集中数量最多的那个类别
			c_k = max(class_D.items(), key=operator.itemgetter(1))[0]
			return node("class", 2, c_k)
		engein_information_gain = {}
		ig = information_gain()

		for c in A.columns:#得到每个特征对类别的信息增益
			engein_information_gain[c] = ig.get_information_gain(D.values.reshape((1, num))[0], A[c].values)[0]
		max_ig = max(engein_information_gain.items(), key=operator.itemgetter(1))#获得信息增益最大的那个
		if max_ig[1] < e:#如果这个信息增益小于阀值e，返回单节点树，标签为类集中数量最大的类别
			# print("less than e")
			c_k = max(class_D.items(), key=operator.itemgetter(1))[0]
			return node("class", 2, c_k)
		dic_max_ig = self.hash(A[max_ig[0]].values)
		T = node(max_ig[0], len(dic_max_ig), label)
		list_A.remove(max_ig[0])
		for key, value in dic_max_ig.items():#根据信息增益最大的特征对类集进行划分
			tmp = pd.DataFrame(columns=(list_A)+["class"])
			for i in range(num):
				if Data_set.loc[i, [max_ig[0]]].values[0] == key:
					tmp = tmp.append(Data_set.loc[i, list_A + ["class"]], ignore_index=True)	
			t_dic = self.hash(tmp["class"].values)
			c_k = max(t_dic.items(), key=operator.itemgetter(1))[0]
			T.children[key] = self.ID3(tmp, e, c_k, list_A)
		return T

	def print_tree(self, T):
		if T == None:
			return
		print("engein:", T.engein)
		print("label", T.label)
		print("value of engein:")
		for key, val in T.children.items():
			if val == None:
				print(key, "->None")
			else:
				print(key, "->", val.engein)
		print()
		for key, val in T.children.items():
			self.print_tree(val)

	def make_dision(self, T, x):
		if T == None:
			return
		if len(T.children) == 0:
			return T.label
		if T.engein == "class":
			return T.label
		if T.engein not in x.columns:
			print("engein not found")
			return
		return self.make_dision(T.children[x[T.engein].values[0]], x)

	def hash(self, l):#统计集合中类别的个数
		dic = {}
		for item in l:
			if item in dic:
				dic[item] += 1
			else:
				dic[item] = 1
		return dic

	def get_leave(self, T):#得到树的叶子
		if T == None:
			return []
		if T.engein == "class":
			return [T]
		l = []
		for key, val in T.children.items():
			l = l + self.get_leave(val)
		return l

	def forecast_err(self, T, Data_set):#得到预测误差
		if T == None:
			return 0
		if T.engein == "class":
			dic = self.hash(Data_set["class"].values)
			# print(dic)
			s = sum(dic.values())
			H = 0
			for key, val in dic.items():
				H -= val/s * np.log(val/s)
			return H
		err = 0
		for key, val in T.children.items():
			tmp = pd.DataFrame(columns=Data_set.columns)
			for i in range(Data_set.shape[0]):
				if Data_set.loc[i, T.engein] == key:
					tmp = tmp.append(Data_set.loc[i, :], ignore_index=True)
			err += self.forecast_err(val, tmp)
		return err

	def get_loss(self, T, alpha, Data_set):#得到损失函数
		leaves = self.get_leave(T)
		return self.forecast_err(T, Data_set) + alpha * len(leaves)

	def pruning(self, T, root, Data_set):#剪枝函数
		if T == None or T.engein == "class":
			return
		for key, val in T.children.items():#对当前节点剪枝前先递归地对子节点剪枝
			self.pruning(val, root, Data_set)
		before_loss = self.get_loss(root, 0.1, Data_set)
		print("before loss", before_loss)
		before_engein = T.engein
		T.engein = "class" #将标签设置为class做懒惰处理，
		after_loss = self.get_loss(root, 0.1, Data_set)
		print("after loss", after_loss)
		if after_loss < before_loss:
			for key in T.children:
				tmp = T.children[key]
				T.children[key] = None
				del tmp
		else:
			T.engein = before_engein
















