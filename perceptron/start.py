import numpy as np
import pandas as pd
from perceptron import Perceptron
from perceptron import Perceptron_antithesis
from data_view import draw

def load_data(filename):
	data_set = np.loadtxt(filename)
	return data_set

def get_data_set_random(num_node):
	data_set = np.random.random((num_node, 3)) * 10
	for i in range(num_node):
		data_set[i,2] = 1 if data_set[i, 0] ** 2 + data_set[i, 1] ** 2 <= 50 else -1
	return data_set


if __name__ == '__main__':
	# data_set = load_data('./data/data.txt')
	data_set = get_data_set_random(30)
	
	#original
	pct = Perceptron(2)
	pct.learn(data_set)
	model = pct.model
	print("positive") if model([2, 2]) == 1 else print("negative")
	parameter = pct.get_parameter()
	print(parameter)
	draw(data_set, parameter)

	#antithesis
	pct2 = Perceptron_antithesis(data_set)
	pct2.learn()
	parameter2 = pct2.get_parameter()
	print(parameter2)
	draw(data_set, parameter2)