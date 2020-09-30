# This python file implements gcn with a triplet loss function

from __future__ import division
from __future__ import print_function

import time
import argparse
import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from models import GCNModelAE_UN
from plots import plot_loss
from triplet_loss import TripletLoss
from utils import load_gene_data, preprocess_graph

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
parser.add_argument('--weight_decay', type=float, default=5e-4, #default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=16,
                    help='Number of clusters.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features = load_gene_data(use_features=True, _adj='homology', _feat='ontology')

# create graph from adjacency matrix
G = nx.from_numpy_matrix(adj.toarray())

# print(adj.shape)
#
# print(G.number_of_edges())
# print(G.number_of_nodes())


adj_label = adj
adj_label = torch.FloatTensor(adj_label.toarray())

adj_train = preprocess_graph(adj)

# Model and optimizer
model = GCNModelAE_UN(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=args.nclass,
                       dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


triplet_loss = TripletLoss(G, False)
indices = []
losses = []
def train(epoch):
    with torch.autograd.set_detect_anomaly(True):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_train)
        # triplet_loss.embeddings = output
        loss = triplet_loss._batch_hard_triplet_loss(output)
        # loss = triplet_loss.get_loss_margin()
        losses.append(loss)
        indices.append(epoch)
        loss.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'time: {:.4f}s'.format(time.time() - t))


for epoch in range(args.epochs):
    train(epoch)

torch.save(model.state_dict(), "models/gcn_hom_onto2")

plot_loss(indices, losses, "Epooch", "Loss", "Epooch vs loss -- gcn ontology")
