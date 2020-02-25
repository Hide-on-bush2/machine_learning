import numpy as np
from bayes import Bayes_classifier

def get_random_data_set(num_node):
	data_set = np.random.randint(2, size=(num_node, 5))
	for i in range(num_node):
		c = np.random.randint(3)
		data_set[i, 4] = c
	return data_set

def get_data_set_from_file(filename):
	data_set = np.loadtxt(filename, dtype='int')
	return data_set


if __name__ == '__main__':
	# data_set = get_random_data_set(10)
	data_set = get_data_set_from_file('./data/data.txt')
	print(data_set)
	new_classifier = Bayes_classifier(data_set, 2, 2)
	new_classifier.learn()
	test_time = 10
	# for i in range(test_time):
	# 	node = np.random.randint(2, size=(4, 1))
	# 	print(new_classifier.classify(node))
	print(new_classifier.classify([1, 0]))
	pri, con = new_classifier.get_model()
	print(pri)
	print(con)