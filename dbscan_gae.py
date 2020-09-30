import torch
import argparse

from cluster_utils import k_means, plot_optimal_clusters, plots_pairwise, plot_heat_map, plot_tsne, db_scan
from ranking import create_ranking
from utils import load_gene_data, preprocess_graph, read_file
import networkx as nx
import pandas as pd
import numpy as np
from models import GCNModelAE

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
parser.add_argument('--saved-model', type=str, default='models/gae', help='Saved model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features = load_gene_data()
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

# Convert to pandas
meta_df = {}
_vars = []
for i in range(args.ndim):
    meta_df[str(i)] = output[:, i]
    _vars.append(str(i))
data_df = pd.DataFrame(meta_df)


# plot_optimal_clusters(data, "DBScan GAE", 'dbscan')
dbs_labels = db_scan(data, 0.1, 5, model_name="models/db_gae_model.pkl")

# Add labels
data_df['DBS-clusters'] = dbs_labels
dbs_unique_clusters = set(dbs_labels)


res = read_file("data/gene/geneset_pairing.csv")
dict = dict((v, k) for k, v in res.items())
data_df['gensets'] = data_df.index.map(dict)
# data_df.set_index('gensets', inplace=True)

sim = data_df[['gensets', 'DBS-clusters']]
grp = data_df[['gensets', 'DBS-clusters']]\
    .sort_values('DBS-clusters')\
    .set_index(['DBS-clusters', 'gensets'], inplace=False)
agg = data_df.groupby('DBS-clusters').agg({'gensets': 'count'})


writer = pd.ExcelWriter('results/dbs_gae.xlsx')
sim.to_excel(writer, sheet_name='simple')
grp.to_excel(writer, sheet_name='grouped')
agg.to_excel(writer, sheet_name='agg')
writer.save()


# Plots here
plots_pairwise(data_df, 'DBS-clusters', _vars, dbs_unique_clusters, "Pairwise Scatter Plot for DBS Clustering")
plot_heat_map(data_df.iloc[:, 0:16])

plot_tsne(data_df.iloc[:, 0:16], data_df['DBS-clusters'],
          "Clustering node embeddings with DBScan Perplexity: {} -- "
          "Number of Iterations: {}", perplexity=45, n_iter=200)

# write to tsv for visualisation
# tsv_df = data_df.iloc[:, 0:17]
# tsv_df.to_csv("tsv/kmeans_gae", sep="\t", index=False)

data = data.numpy()
create_ranking('results/ranking_gae.xlsx', data=data, _ids=[834, 18840, 271959, 1051], )
