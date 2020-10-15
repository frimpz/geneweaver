# This file contains plots for visualizing data.
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Pairwise Scatter plot for features
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance


def pairwise_scatter(data, hue, x_vars, y_vars, title="", marker=None):
    g = sns.pairplot(data, markers=marker, hue=hue, height=3, x_vars=x_vars, y_vars=y_vars)
    g.fig.suptitle(title)
    plt.show()


# Pairwise Scatter plot for features
def heat_map(data, title):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f', linewidths=.05)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    hm.set_title(title, fontsize=14)
    plt.show()


def tsne_plot(x, y, clusters, title=""):
    df_plot = pd.DataFrame()
    df_plot['x'] = x
    df_plot['y'] = y
    df_plot['clusters'] = clusters
    # muted set1 hls
    color_palette = sns.color_palette("muted", len(set(clusters)))
    plt.figure(figsize=(16, 10))

    color_patches =[]
    for i, label in enumerate(set(clusters)):
        # add data points
        plt.scatter(x=df_plot.loc[df_plot['clusters'] == label, 'x'],
                    y=df_plot.loc[df_plot['clusters'] == label, 'y'],
                    color=color_palette[i],
                    alpha=1)

        # add label
        plt.annotate(label,
                     df_plot.loc[df_plot['clusters'] == label, ['x', 'y']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=20, weight='bold',
                     color='white',
                     backgroundcolor=color_palette[i])

        color_patches.append(mpatches.Patch(color_palette[i], label=label, facecolor=color_palette[i]))

    # sns.scatterplot(
    #     x="x",
    #     y="y",
    #     hue=clusters,
    #     palette=color_palette,
    #     data=df_plot,
    #     legend="full",
    #     alpha=0.8
    # )
    # plt.annotate()
    plt.legend(loc="lower right", handles=color_patches)
    plt.title(title)
    plt.show()


def k_meansElbow(data, type):
    Sum_of_squared_distances = []
    _min = 3
    _max = 40
    stp = 5
    K = range(_min, _max)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K, Sum_of_squared_distances, curve='convex', direction='decreasing')
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k --- ' + type)
    plt.text(_max - 2 * stp, max(Sum_of_squared_distances), 'k = %d' % kn.knee)
    plt.show()


def silhouettevisual(model, X, graph):
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick', title=" Silhouette Plot of KMeans Clustering for "+ graph)
    visualizer.fit(X)
    visualizer.show()


def cluster_distances(model, X, graph):
    visualizer = InterclusterDistance(model, legend=True, legend_loc='upper left', title=" KMeans Intercluster Distance Map for "+ graph)
    visualizer.fit(X)
    visualizer.show()


def opt_dbScan(data, type):
    neighbour = NearestNeighbors(n_neighbors=2)
    nbrs = neighbour.fit(data)
    dist, indicies = nbrs.kneighbors(data, return_distance=True)
    dist = np.sort(dist, axis=0)
    dist = dist[:, 1]
    plt.plot(dist)
    plt.title('Optimal DBScan Plot --- '+ type)
    plt.show()


def plot_loss(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def auc_roc(emb, adj_orig, edges_pos, edges_neg, roc_score, title):
    lw = 2

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    ##########################confirm
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    fpr, tpr, tresholds = roc_curve(labels_all, preds_all)

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic '+ title)
    plt.legend(loc="lower right")
    plt.show()

