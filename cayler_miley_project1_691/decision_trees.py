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
def DT_train_binary(X, Y, max_depth):
	depth_count = 0

	if(max_depth < 0):
		# iterate until maximum accuracy reached
		max_depth = min(len(X[0]), len(X)-1)
		print("Max Depth: ", max_depth)

	used_features = []

	tree = choose_best_tree(X, Y, used_features, depth_count, max_depth)

	print(tree)

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


def choose_best_tree(X, Y, feature_indices, current_depth, max_depth, is_binary=True):
	if is_binary:
		threshold = 0.5

	print("Y", Y)

	if same_labels(Y):
		print("same labels")
		return {"accuracy": 1.0, "output": Y[0][0]}
	elif same_features(X):
		print("same features")
		label, count = highest_count(Y)
		return {"accuracy": count/len(Y), "output": highest_count(Y)[0]}
	elif current_depth == max_depth:
		print("reached max depth")
		label, count = highest_count(Y)
		return {"accuracy": count/len(Y), "output": highest_count(Y)[0]}

	best_acc = 0

	if X is None:
		print("return None")
		return None

	for f_index in range(len(X[0])):
		if f_index in feature_indices:
			pass
		else:
			(accuracy, leq_indices, gre_indices) = get_accuracy(X, Y, f_index, threshold)
			if(accuracy > best_acc):
				feature_index = f_index
				best_acc = accuracy

	feature_indices.append(feature_index)

	print("leq indices", leq_indices)
	lower_tree = choose_best_tree(X[leq_indices], Y[leq_indices], feature_indices, current_depth+1, max_depth, is_binary=is_binary)

	print("gre indices", gre_indices)
	upper_tree = choose_best_tree(X[gre_indices], Y[gre_indices], feature_indices, current_depth+1, max_depth, is_binary=is_binary)

	tree = {
		"accuracy": best_acc,
		"feature": feature_index, 
		(feature_index, 0): lower_tree,
		(feature_index, 1): upper_tree
	}

	return tree

def get_accuracy(samples, labels, feature, threshold):
	below = []
	above = []
	leq_indices = []
	gre_indices = []
	correct = {}

	for s_index, sample in enumerate(samples):
		if sample[feature] <= threshold:
			below.append(labels[s_index])
			leq_indices.append(s_index)
		else:
			above.append(labels[s_index])
			gre_indices.append(s_index)
	correct["leq"] = [below.count(0), below.count(1)]
	correct["gre"] = [above.count(0), above.count(1)]
	#print(correct)

	accuracy = max(correct["leq"][0] + correct["gre"][1], correct["leq"][1] + correct["gre"][0])/len(samples) 
	#print(accuracy)

	return (accuracy, leq_indices, gre_indices)


def same_labels(vec):
	for val in vec:
		if not np.array_equal(val, vec[0]):
			return False
	return True


def same_features(X):
	for i in X:
		for j in X:
			if not np.array_equal(X[i], X[j]):
				return False
	return True


def highest_count(labels):
	unique, counts = np.unique(labels, return_counts=True)

	if not len(labels):
		return [None], 0

	print("Labels", labels)
	max_count = 0

	for index, count in enumerate(counts):
		if count > max_count:
			max_count = count
			label = unique[index]

	return label, max_count


def main():
	DT_train_binary(X_train_1, Y_train_1, 3)
	DT_train_binary(X_train_2, Y_train_2, 3)


if __name__ == '__main__':
	main()