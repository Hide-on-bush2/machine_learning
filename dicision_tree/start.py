import pandas as pd
import numpy as np
from dicision_tree import Dicision_tree

def get_random_data(data_size):
	dic = {"year" : np.random.randint(3, size=(data_size)),
			"work" : np.random.randint(2, size=(data_size)),
			"house" : np.random.randint(2, size=(data_size)),
			"credit" : np.random.randint(3, size=(data_size)),
			"class" : np.random.randint(2, size=(data_size))}
	data_set = pd.DataFrame(dic, index=range(data_size))
	return data_set

if __name__ == '__main__':
	data_set = get_random_data(15)
	print(data_set)