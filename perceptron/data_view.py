import matplotlib.pyplot as plt

def draw(data_set, parameter):
	x_1 = data_set[:, 0]
	x_2 = data_set[:, 1]
	color = data_set[:, 2]
	fig = plt.figure()
	plt.scatter(x_1, x_2, c=color)

	w, b = parameter[0], parameter[1]

	#draw a classifying line
	#x_1 = 0
	t_x_2 = (-b) / w[1]
	#x_2 = 0
	t_x_1 = (-b) / w[0]

	plt.plot([0, t_x_1], [t_x_2, 0])
	plt.show()
