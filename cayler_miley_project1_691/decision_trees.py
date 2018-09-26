import numpy as np 


## Decision Trees
# Training Set 1:
X_train_1 = np.array([[0,1], [0,0], [1,0], [0,0], [1,1]])
Y_train_1 = np.array([[1], [0], [0], [0], [1]])
# Validation Set 1:
X_val_1 = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_val_1 = np.array([[0], [1], [0], [1]])
# Testing Set 1:
X_test_1 = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_test_1 = np.array([[1], [1], [0], [1]])
# Training Set 2:
X_train_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1],
                      [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
Y_train_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
# Validation Set 2:
X_val_2 = np.array([[1,0,0,0], [0,0,1,1], [1,1,0,1],
                    [1,1,0,0], [1,0,0,1], [0,1,0,0]])
Y_val_2 = np.array([[0], [0], [1], [0], [1], [1]])
# Testing Set 2:
X_test_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1],
                     [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
Y_test_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])
 
X_real = np.array([[4.8,3.4,1.9,0.2], [5.0,3.0,1.6,1.2], [5.0,3.4,1.6,0.2],
                   [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.2], [4.7,3.2,1.6,0.2],
                   [4.8,3.1,1.6,0.2], [5.4,3.4,1.5,0.4], [7.0,3.2,4.7,1.4],
                   [6.4,3.2,4.7,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4.0,1.3],
                   [6.5,2.8,4.6,1.5], [5.7,2.8,4.5,1.3], [6.3,3.3,4.7,1.6],
                   [4.9,2.4,3.3,1.0]])
Y_real = np.array([[1],[1],[1],[1],[1],[1],[1],[1],
                   [0],[0],[0],[0],[0],[0],[0],[0]])


'''
Basic Requirements:
'''
def DT_train_binary(X,Y,max_depth):
	if(max_depth < 0):
		# iterate until maximum accuracy reached
		pass

	for feature in X[0]:
		print("feature")

	for i in range(max_depth):
		# generate decision trees
		pass


def DT_test_binary(X,Y,DT):
	pass


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
	pass


'''
691 (Graduate) Requirements:
'''
def DT_train_real(X,Y,max_depth):
	pass


def DT_test_real(X,Y,DT):
	pass


def DT_train_real_best(X_train,Y_train,X_val,Y_val):
	pass


def main():
	print("decision trees")
	DT_train_binary(X_train_1, Y_train_1, 3)


if __name__ == '__main__':
	main()