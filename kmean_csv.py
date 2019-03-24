from random import choice
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import ceil, floor
import csv
from sklearn.datasets import make_blobs

# randomly generates centroids without repeats, takes the number of clusters
# and the data points as inputs, returns a list of the coordinates of the
# centroids' initial positions

def gen_clusters(k, data_points):

    x_vals = [i[0] for i in data_points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_vals = [i[1] for i in data_points]
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = list(range(ceil(x_min), floor(x_max + 1)))
    y_range = list(range(ceil(y_min), floor(y_max + 1)))

    cluster_locations = []
    for j in range(k):
        x, y = choice(x_range), choice(y_range)
        cluster_locations.append([x, y])
        x_range.remove(x)
        y_range.remove(y)

    return(cluster_locations)

# associates data points with their nearest cluster, takes the point locations
# and the cluster locations as inputs, returns the index of the closest cluster
# for each point

def nearest_cluster(data_points, cluster_locations):
    distances = []

    for point in data_points:
        temp_distances = []
        for cluster in cluster_locations:
            temp_distances.append( ( (point[0] - cluster[0]) ** 2 + (point[1] - cluster[1]) ** 2 ) ** 0.5 )
        distances.append(temp_distances.index(min(temp_distances)))

    return(distances)

# returns all points associated with a particular cluster
def associate_points(cluster_associations, data_points, cluster_number):
    return([data_points[index] for index, value in enumerate(cluster_associations) if value == cluster_number])

# finds the mean of the points associated with a cluster
def point_mean(cluster_associations, data_points, cluster_number):

   #try-except in the case that no points associate with a centroid, which returns a divide by zero since len(new_x) and len(new_y) are 0
    try:
        #which points are associated with cluster x?
        associated = associate_points(cluster_associations, data_points, cluster_number)

        #where is the mean of those points?
        new_x = [i[0] for i in associated]
        mean_x = sum(new_x) / len(new_x)

        new_y = [i[1] for i in associated]
        mean_y = sum(new_y) / len(new_y)

        new_mean_location = [mean_x, mean_y]

        return(new_mean_location)

    except:
        pass

# initialize the algorithm by setting the number of clusters, generating
# datapoints, generating clusters, and running the first iteration of the
# point-cluster association 

data_source = 'random'
if data_source == 'csv':
    with open('data.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        data_strings = list(reader)
        dataset = [list(map(float, i)) for i in data_strings]
else:
    X, y = make_blobs(n_samples=1000, n_features=2, centers=3)
    dataset = list(map(list, X))

k_clusters = 3
initial_clusters = gen_clusters(k_clusters, dataset)
associations = nearest_cluster(dataset, initial_clusters)
new_clusters = list(initial_clusters)

# associate points with the nearest cluster, find mean of points associated
# with that cluster, and move the cluster to the new mean, and repeat until the
# cluster no longer moves, meaning that the clustering is complete

for i in range(200):
    associations = nearest_cluster(dataset, new_clusters)
    new_clusters = [point_mean(associations, dataset, cluster_num) for cluster_num in range(k_clusters)]

final_associations = [associate_points(associations, dataset, cluster_num) for cluster_num in range(k_clusters)]
rounded_new_clusters = [list(map(lambda x: "%.4f" %x, i)) for i in sorted(new_clusters)]

# sklearn kmeans for comparison
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k_clusters)
kmeans.fit_transform(dataset)
rounded_cluster_centers = [list(map(lambda x: "%.4f" %x, i)) for i in sorted(kmeans.cluster_centers_.tolist())]

print(rounded_new_clusters)
print(rounded_cluster_centers)

# plots two graphs, one with the initial clusters and the unclassified data
# points and another with the moved clusters and their associated data points
# in different colors

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter([i[0] for i in dataset], [i[1] for i in dataset], color='c')
ax1.scatter([i[0] for i in initial_clusters], [i[1] for i in initial_clusters], color='k')
ax1.set_title("Initial Clustering")

colors = ['c', 'r', 'y', 'b', 'g', 'm']
for i in range(k_clusters):
    ax2.scatter([j[0] for j in final_associations[i]], [j[1] for j in final_associations[i]], color=colors[i])

ax2.scatter([i[0] for i in new_clusters], [i[1] for i in new_clusters], label="new_centroids", color='k')
ax2.set_title("Final Clustering")

plt.show()

