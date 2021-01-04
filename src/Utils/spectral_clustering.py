import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


def sp_clustering(img):
    graph = image.img_to_graph(img)

    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    graph.data = np.exp(-graph.data / graph.data.std())

    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    labels = spectral_clustering(graph, n_clusters=64, eigen_solver='arpack')

    plt.matshow(img)
    plt.matshow(labels)


img = io.imread(
    '/media/ali/ssd/PNG201709_small/Foot/2012/01/28385--42266--2012-01-03-09.20.31--unknown--Tfrontaldx.png')
sp_clustering(img)
