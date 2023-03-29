# DBSCAN clustering with Python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers,cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X) # standardize the dataset
xx, yy = zip(*X)
db = DBSCAN(eps=0.3, min_samples=1000).fit(X) # Set up parameters
core_samples = db.core_sample_indices_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
n_clusters_ = len(set(labels_true))-(1 if -1 in labels_true else 0) # Nr of clusters
labels = db.labels_
outliers = X[labels == -1] # find the outliers
cluster1 = X[labels == 0] # Get the contents of each cluster
cluster2 = X[labels == 1]
cluster3 = X[labels == 2]
unique_labels = set(labels)
colors = ['y', 'b', 'g', 'r']

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k',markersize=6)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.title('number of clusters: %d' %n_clusters_)
plt.show()