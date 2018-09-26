import numpy as np 
import random
import sys
import matplotlib.pyplot as plt

## K-Means
X_set = np.array([[1,0],[7,4],[9,6],[2,1],[4,8],[0,3],[13,5],
              [6,8],[7,3],[3,6],[2,1],[8,3],[10,2],[3,5],
              [5,1],[1,9],[10,3],[4,1],[6,6],[2,2]])

threshold = 0.1


'''
Basic Requirements:
'''
def K_Means(X, K):
	# choose random samples to initialize cluster centers
	centers = init_cluster_centers(X, K)
	updated_centers = []

	updated_clusters = compute_clusters(X, centers)
	stale_clusters = []

	while not converged(stale_clusters, updated_clusters):
		stale_clusters = updated_clusters
		centers = compute_cluster_centers(X, updated_clusters)
		updated_clusters = compute_clusters(X, centers)

	return np.sort(centers, axis=0)


'''
691 (Graduate) Requirements:
'''
def K_Means_better(X, K):
	center_counts = []
	centers = []
	error = sys.maxsize
	while centers_converged(center_counts) == -1:
		new_center = K_Means(X, K)
		centers.append(new_center)
		error = window_error(centers)
		center_counts = update_counts(center_counts, new_center)
		centers_converged(center_counts)

	mean_centers = np.mean(centers, axis=0)
	centers = np.asarray(centers)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	plt.scatter(X_set[:,0], X_set[:,1], s=1, c = 'b', label="Training")
	plt.scatter(centers[:,0,0], centers[:,0,1], s=1, c='r', label="Cluster 1")
	plt.scatter(centers[:,1,0], centers[:,1,1], s=1, c='g', label="Cluster 2")
	plt.scatter(centers[:,2,0], centers[:,2,1], s=1, c='m', label="Cluster 3")
	plt.show()

	return mean_centers


def init_cluster_centers(X, K, use_sample=True):
	centers = []

	while len(centers) < K:
		new_center = random.choice(X)
		if not in_list(centers, new_center):
			centers.append(new_center)

	return centers

def compute_clusters(X, cluster_centers):
	clusters = [[] for center in cluster_centers]

	for index, point in enumerate(X):
		min_dist = sys.maxsize
		closest = 0
		for num, center in enumerate(cluster_centers):
			dist = L2_norm(center, point)
			if dist < min_dist:
				closest = num
				min_dist = dist
		clusters[closest].append(index)

	return clusters

def compute_cluster_centers(samples, clusters):
	centers = []
	for cluster in clusters:
		new_center = np.mean(samples[cluster], axis=0)
		centers.append(new_center)

	return centers


def L2_norm(point1, point2):
	return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2


def in_list(arr, element):
	if(len(arr) > 0):
		for arr_element in arr:
			if all(arr_element == element):
				return True

	return False


def converged(old_clusters, new_clusters):
	old_clusters = np.sort(old_clusters, axis=0)
	new_clusters = np.sort(new_clusters, axis=0)

	if np.array_equal(old_clusters, new_clusters):
		return True

	return False


def centers_converged(center_counts):
	if not center_counts:
		return -1
	total_count = 0
	max_count = 0
	max_index = -1

	for index, (center, count) in enumerate(center_counts):
		total_count += count
		if count > max_count:
			max_index = index
			max_count = count
		
	# must run more than 10 trials
	if total_count < 10:
		return -1

	print(max_count/total_count, " occurence")
	if max_count/total_count >= 0.5:
		return max_index

	# stop running beyond 1000 trials
	if total_count > 999:
		return max_index

	return -1


def update_counts(center_counts, new_center):
	total_count = 0

	if not center_counts:
		total_count += 1
		center_counts.append([new_center, 1])

	for index, (center, count) in enumerate(center_counts):
		total_count += count
		if np.array_equal(center, new_center):
			center_counts[index][1] = center_counts[index][1] + 1
			return center_counts

	center_counts.append([new_center, 1])

	return center_counts


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def window_error(list_of_centers):
	mean_centers = np.mean(list_of_centers, axis=0)
	error = 0

	for centers in list_of_centers:
		error += rmse(centers, mean_centers)

	window_error = error/(len(list_of_centers))

	return window_error


def main():
	K_Means(X_set, 3)
	K_Means_better(X_set, 5)


if __name__ == '__main__':
	main()