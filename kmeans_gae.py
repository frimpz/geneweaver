import pickle

import torch
import argparse

from cluster_utils import k_means, plot_optimal_clusters, plots_pairwise, plot_heat_map, plot_tsne
from ranking import create_ranking, create_ranking_
from utils import load_gene_data, preprocess_graph, read_file, write_file
import networkx as nx
import pandas as pd
import numpy as np
from models import GCNModelAE
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Saved models: gae_model


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--ndim', type=int, default=16,
                    help='Number of dimension.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--saved-model', type=str, default='models/gae_hom_only', help='Saved model')
parser.add_argument('--title', type=str, default=' -- GAE -- homology -- homology', help='graph form')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features = load_gene_data(use_features=True, _adj='homology', _feat='homology')
G = nx.from_numpy_matrix(adj.toarray())
adj_train = preprocess_graph(adj)


model = GCNModelAE(nfeat=features.shape[1], nhid=args.hidden
                   , nclass=args.ndim, dropout=args.dropout)
model.load_state_dict(torch.load(args.saved_model))
model.eval()


model(features, adj_train)
output = model.mu.data

# Normalize the output data
data = output
# scaler = StandardScaler().fit(output)
# data = scaler.transform(output)

# # Convert to pandas
# meta_df = {}
# _vars = []
# for i in range(args.ndim):
#     meta_df[str(i)] = output[:, i]
#     _vars.append(str(i))
# data_df = pd.DataFrame(meta_df)
#
#
# # plot_optimal_clusters(data, "K-means GAE" + args.title, 'kmeans')
# kmeans_labels = k_means(data, 6, model_name="models/kmeans_gae_model.pkl")
#
# # Add labels
# data_df['KMeans-clusters'] = kmeans_labels
# kmeans_unique_clusters = set(kmeans_labels)


# # compute davies and silhoutte scores
# scores = {}
# scores['dav_score'] = davies_bouldin_score(data, kmeans_labels)
# scores['sil_score'] = silhouette_score(data, kmeans_labels)
# write_file('results/cluster_scores_gae.csv', scores)
#
#
# res = read_file("data/ms-project/geneset_pairing.csv")
# dict = dict((v, k) for k, v in res.items())
# data_df['gensets'] = data_df.index.map(dict)
# # data_df.set_index('gensets', inplace=True)
#
# sim = data_df[['gensets', 'KMeans-clusters']]
# grp = data_df[['gensets', 'KMeans-clusters']]\
#     .sort_values('KMeans-clusters')\
#     .set_index(['KMeans-clusters', 'gensets'], inplace=False)
# agg = data_df.groupby('KMeans-clusters').agg({'gensets': 'count'})
#
#
# writer = pd.ExcelWriter('results/kmeans_gae_onto_onto.xlsx')
# sim.to_excel(writer, sheet_name='simple')
# grp.to_excel(writer, sheet_name='grouped')
# agg.to_excel(writer, sheet_name='agg')
# writer.save()
#
#
# Plots here
# plots_pairwise(data_df, 'KMeans-clusters', _vars, kmeans_unique_clusters,
#                 "Pairwise Scatter Plot for KMeans Clustering" + args.title)
# plot_heat_map(data_df.iloc[:, 0:16], "Correlation matrix for Node Embeddings " + args.title)
#
# plot_tsne(data_df.iloc[:, 0:16], data_df['KMeans-clusters'],
#           "Clustering node embeddings with KMeans Perplexity: {} -- "
#           "Number of Iterations: {}" + args.title, perplexity=50, n_iter=1000)

# write to tsv for visualisation
# tsv_df = data_df.iloc[:, 0:17]
# tsv_df.to_csv("tsv/kmeans_gae", sep="\t", index=False)

data = data.numpy()
create_ranking_('ranking_results/'+args.title+'.csv', data=data)
# create_ranking('results/ranking_gae.xlsx', data=data, _ids=[834, 18840, 271959, 1051])


# with open('results/samples.pkl', 'rb') as fp:
#     samples = pickle.load(fp)

# create_ranking('ranking_results/'+args.title+'.csv', data=data, file_type='csv')
