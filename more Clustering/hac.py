#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from datasets import (
    circles,
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances
)

X = circles()

hac = AgglomerativeClustering(n_clusters=3, linkage='single')
hac.fit(X)

plt.scatter(X[:,0], X[:,1], c=hac.labels_)
plt.show()
