from random import choice
from matplotlib import pyplot as plt
from file import dataset

# randomly generates centroids without repeats, takes the number of clusters
# and the data points as inputs, returns a list of the coordinates of the
# centroids' initial positions

def gen_clusters(k, data_points):

    x_vals = [i[0] for i in data_points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_vals = [i[1] for i in data_points]
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = list(range(x_min, x_max + 1))
    y_range = list(range(y_min, y_max + 1))

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

k_clusters = 4
initial_clusters = gen_clusters(k_clusters, dataset)
associations = nearest_cluster(dataset, initial_clusters)
new_clusters = list(initial_clusters)

# associate points with the nearest cluster, find mean of points associated
# with that cluster, and move the cluster to the new mean, and repeat until the
# cluster no longer moves, meaning that the clustering is complete

while new_clusters != [point_mean(associations, dataset, cluster_num) for cluster_num in range(k_clusters)]:
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
