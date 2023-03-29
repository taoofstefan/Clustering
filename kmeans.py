#K-Means clustering with Python
import pandas as pd # For reading datasets
import numpy as np # For computations
import matplotlib.pyplot as plt # For visualization
from pandas import DataFrame # For creating data frame
from sklearn.cluster import KMeans

x = [np.random.randint(0,500) for x in range(350)]
y = [np.random.randint(0,500) for y in range(350)]

Data={
'x': x,
'y': y
}

df = DataFrame(Data,columns=['x','y'])
# Create and fit the KMeans model
kmeans = KMeans(n_clusters=5).fit(df)
# Find the centroids of the clusters
centroids = kmeans.cluster_centers_
# Get the associated cluster for each data record
kmeans.labels_
# Display the clusters contents and their centroids
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50,
alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()