"""
Cluster trees in random forest based on their selected features.
Input: 
N: number of trees
m: number of features
k: predefined number of clusters
"""

import argparse
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    """ Parse the input arguments """

    parser = argparse.ArgumentParser(description='Tree clustering on selected features.')
    parser.add_argument('--trees', dest='N', default=10000, type =int, help='The number of trees')
    parser.add_argument('--features', dest='m', default=10, type =int, help='The number of features')
    parser.add_argument('--cluster', dest='n_clusters', default=2, type =int, help='preset cluster numbers')
    parser.add_argument('--debug', dest='DEBUG', default=True, type =bool, help='Enable main parameter print mode or not.')

    args = parser.parse_args()

    if args.DEBUG:
        print("input parameters:" +str(args))
    return args


def feature_generator(N, m, p=0.6):
    """ 
    N: Data dimension/num of trees
    m: feature dimension
    p: proportion of one/feature selected
    """
    data = np.zeros((N,m))
    for i in range(N):
        nums = np.random.choice([0, 1], size=m, p=[1-p, p])
        data[i,:] = nums
    return data


if __name__ == '__main__':

    # Grab the input parameters
    args = parse_args()

    # generate the simulated feature data as the input of the clustering analysis
    tree_features = feature_generator(args.N, args.m)

    if args.DEBUG:
        print('The simulated tree feature selection data:\n' + str(tree_features))

    # Cluster the tree feature selection matrix
    kmeans = KMeans(n_clusters= args.n_clusters, random_state=0).fit(tree_features)
    
    labels = kmeans.labels_

    print(labels)

    fig = plt.figure(figsize= (12,12))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(tree_features[:, 0], tree_features[:, 1], tree_features[:, 2], c=labels)
    #plt.title("clustering analysis")
    plt.show()




