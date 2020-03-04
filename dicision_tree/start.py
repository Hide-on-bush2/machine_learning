import pandas as pd
import numpy as np
import operator
from dicision_tree import Dicision_tree, node
from information_gain import information_gain
from cart_regression import CartRegression
from cart_classifision import CartClassifision

def get_random_data(data_size):
	dic = {"year" : np.random.randint(3, size=(data_size)),
			"work" : np.random.randint(2, size=(data_size)),
			"house" : np.random.randint(2, size=(data_size)),
			"credit" : np.random.randint(3, size=(data_size)),
			"class" : np.random.randint(2, size=(data_size))}
	data_set = pd.DataFrame(dic, index=range(data_size))
	return data_set

def get_data_from_file(filename, columns):
	return pd.read_table(filename, sep=' ', names=columns)

if __name__ == '__main__':
	
	# data_set = get_random_data(15)
	data_set = get_data_from_file('./Data/data.txt', ["year", "work", "house", "credit", "class"])
	# print(data_set)
	engein = data_set.loc[:, ["year", "work", "house", "credit"]]
	# print("engein vector:")
	# print(engein)
	_class = data_set.loc[:, ["class"]]
	# print("class:")
	# print(_class)
	# ig = information_gain()#测试信息增益模块
	# print("test experience in entropy")
	# for c in data_set.columns:
	# 	print(c)
	# 	print(ig.get_entropy(data_set[c].values))

	# print("test simulate the information gain:")
	# dic = {}
	# D = data_set["class"].values
	# for c in engein.columns:
	# 	print(c)
	# 	A = engein[c].values
	# 	condition_entropy = ig.get_information_gain(D, A)
	# 	dic[c] = condition_entropy[0]
	# 	print(condition_entropy)
	# print("max:")
	# print(max(dic.items(), key=operator.itemgetter(1)))
	#测试ID3决策树
	# T = Dicision_tree()
	# dic_class = T.hash(data_set["class"].values)#用D中数量最多的类别作为根节点的标签
	# dicition_tree = T.ID3(data_set, 0.1, max(dic_class.items(), key=operator.itemgetter(1))[0], ["year", "work", "house", "credit"])#生成决策树
	# T.print_tree(dicition_tree)
	# x_dic = {"year" : np.random.randint(3),#用决策树进行决策
	# 		"work" : np.random.randint(2),
	# 		"house" : np.random.randint(2),
	# 		"credit" : np.random.randint(3)}
	# x = pd.DataFrame(x_dic, index=range(1))

	# print(x)
	# print(T.make_dision(dicition_tree, x))
	#测试剪枝
	# T.pruning(dicition_tree, dicition_tree, data_set)#测试剪枝
	# print("after pruning")
	# T.print_tree(dicition_tree)
	#测试cart回归树
	# _cart = CartRegression()
	# aver_class = np.average(data_set["class"].values)
	# T = _cart.cart_regres(data_set, ["year", "work", "house", "credit"], "class", aver_class)
	# _cart.print_tree(T)

	# x_dic = {"year" : np.random.randint(3),#用cart回归树进行回归分析
	# 		"work" : np.random.randint(2),
	# 		"house" : np.random.randint(2),
	# 		"credit" : np.random.randint(3)}
	# x = pd.DataFrame(x_dic, index=range(1))

	# print(x)
	# val = _cart.get_regression_value(T, x)
	# print(val)

	#测试cart生成树
	cart_ = CartClassifision()
	T = cart_.cart_classify(data_set, ["year", "work", "house", "credit"], "class")
	cart_.print_tree(T)
	forest = cart_.pruning(T, data_set, ["year", "work", "house", "credit"], "class")
	for alpha, tree in forest.items():
		print(alpha, "->")
		cart_.print_tree(tree)













