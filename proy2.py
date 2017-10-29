# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


__author__ = "Josué Fabricio Urbina González"
# Each example is a 28x28 grayscale image, associated with a label from 10 classes
# Each row is the image of 784 values. Each dimension is a position in the image
# Data from https://github.com/zalandoresearch/fashion-mnist

# Visual
# Random state.
RS = 102030

data = input_data.read_data_sets('Data/P2')

tmp = data.train.images
tmp_labels = data.train.labels

# t-SNE algorithm
# Reorder the data points according to the classes.
X = np.vstack([tmp[tmp_labels == i]
               for i in range(10)])
y = np.hstack([tmp_labels[tmp_labels == i]
               for i in range(10)])

set = set()
while len(set) < 6000:
    set.add(np.random.randint(0, len(tmp)))
set = sorted(list(set))
train = X[list(set), :]
train_labels = y[list(set)]

proj = TSNE(random_state=RS, perplexity=50, verbose=1).fit_transform(train)


def scatter(x):
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    return f, ax, sc


scatter(proj)
plt.savefig('fashion_tsne-generated.png', dpi=120)


# Clustering
# PCA Dimensionality reduction
pca = PCA(n_components=300)
proj = pca.fit_transform(train)

# Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=10)
cluster.fit(proj)


print(cluster.labels_)





