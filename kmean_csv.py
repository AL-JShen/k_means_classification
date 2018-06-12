from random import choice
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import ceil, floor
import csv

# grab column headers from csv

with open('data.csv', newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)

# read data from csv and separate into columns

data = pd.read_csv('data.csv')
col1 = list(data[headers[0]].values)
col2 = list(data[headers[1]].values)

# wow thats a lot of listing. basically just formats the data so that its in
# the form of a list of lists

dataset = list(map(list, list(zip(col1, col2))))

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

k_clusters = 3
initial_clusters = gen_clusters(k_clusters, dataset)
associations = nearest_cluster(dataset, initial_clusters)
new_clusters = list(initial_clusters)

# associate points with the nearest cluster, find mean of points associated
# with that cluster, and move the cluster to the new mean, and repeat until the
# cluster no longer moves, meaning that the clustering is complete

# while loop doesnt work for some reason, idea behind it was to keep going
# until it would no longer move, but it works with a for loop and the
# performance isnt that bad, 100 loops takes like 1~ second

#while new_clusters != [point_mean(associations, dataset, cluster_num) for cluster_num in range(k_clusters)]:
for i in range(100):
    associations = nearest_cluster(dataset, new_clusters)
    new_clusters = [point_mean(associations, dataset, cluster_num) for cluster_num in range(k_clusters)]


final_associations = [associate_points(associations, dataset, cluster_num) for cluster_num in range(k_clusters)]

# plots two graphs, one with the initial clusters and the unclassified data
# points and another with the moved clusters and their associated data points
# in different colors

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter([i[0] for i in dataset], [i[1] for i in dataset], color='c')
ax1.scatter([i[0] for i in initial_clusters], [i[1] for i in initial_clusters], color='k')
ax1.set_title("Initial Clustering")

# sorry this is some really ugly code but i cant get the colors and labels
# working when i try to plot the points with a for loop and i need the colors
# to work this is a temporary solution to dynamically color the different
# clustered points

if k_clusters == 2:
    ax2.scatter([i[0] for i in final_associations[0]], [i[1] for i in final_associations[0]], color='c')
    ax2.scatter([i[0] for i in final_associations[1]], [i[1] for i in final_associations[1]], color='r')
elif k_clusters == 3:
    ax2.scatter([i[0] for i in final_associations[0]], [i[1] for i in final_associations[0]], color='c')
    ax2.scatter([i[0] for i in final_associations[1]], [i[1] for i in final_associations[1]], color='r')
    ax2.scatter([i[0] for i in final_associations[2]], [i[1] for i in final_associations[2]], color='y')
elif k_clusters == 4:
    ax2.scatter([i[0] for i in final_associations[0]], [i[1] for i in final_associations[0]], color='c')
    ax2.scatter([i[0] for i in final_associations[1]], [i[1] for i in final_associations[1]], color='r')
    ax2.scatter([i[0] for i in final_associations[2]], [i[1] for i in final_associations[2]], color='y')
    ax2.scatter([i[0] for i in final_associations[3]], [i[1] for i in final_associations[3]], color='b')
elif k_clusters == 5:
    ax2.scatter([i[0] for i in final_associations[0]], [i[1] for i in final_associations[0]], color='c')
    ax2.scatter([i[0] for i in final_associations[1]], [i[1] for i in final_associations[1]], color='r')
    ax2.scatter([i[0] for i in final_associations[2]], [i[1] for i in final_associations[2]], color='y')
    ax2.scatter([i[0] for i in final_associations[3]], [i[1] for i in final_associations[3]], color='b')
    ax2.scatter([i[0] for i in final_associations[4]], [i[1] for i in final_associations[4]], color='g')
elif k_clusters == 6:
    ax2.scatter([i[0] for i in final_associations[0]], [i[1] for i in final_associations[0]], color='c')
    ax2.scatter([i[0] for i in final_associations[1]], [i[1] for i in final_associations[1]], color='r')
    ax2.scatter([i[0] for i in final_associations[2]], [i[1] for i in final_associations[2]], color='y')
    ax2.scatter([i[0] for i in final_associations[3]], [i[1] for i in final_associations[3]], color='b')
    ax2.scatter([i[0] for i in final_associations[4]], [i[1] for i in final_associations[4]], color='g')
    ax2.scatter([i[0] for i in final_associations[5]], [i[1] for i in final_associations[5]], color='m')

ax2.scatter([i[0] for i in new_clusters], [i[1] for i in new_clusters], label="new_centroids", color='k')
ax2.set_title("Final Clustering")

plt.show()
