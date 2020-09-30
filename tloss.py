import torch
import torch.nn.functional as F

from utils import my_cross_entropy_with_logits
import random

class TLoss():
    def __init__(self, G, embeddings= None):
        super(TLoss, self).__init__()
        self.Q = 10
        self.MARGIN = 3
        self.NUM_WALKS = 6
        self.POS_WALK_LEN = 1
        self.NEG_WALK_LEN = 15
        self.NUM_NEG = 6

        self.adj_lists = self.get_adjacency_list(G)
        self.graph = G
        self.embeddings = embeddings
        self.pos = {}
        self.negs = {}

        self.create_positives(3)
        self.create_neg()

    def get_adjacency_list(self, G):
        adj_list = {}
        for i in G.adjacency():
            adj_list[i[0]] = [int(x) for x in i[1] if int(x) is not i[0]]
        return adj_list

    def cal_sim(self, node_embedding, pos_neg_emb):
        pos_score = F.cosine_similarity(node_embedding, pos_neg_emb)
        pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)
        return pos_score

    def calc_euclidean(self, x_1, x_2):
        return (x_1-x_2).pow(2).sum(1)

    def get_loss_margin(self):
        nodes_score = []
        nodes = list(self.adj_lists.keys())
        for node in nodes:
            pos_nodes_ind = list(self.pos[node])
            node_pos = [node] * len(pos_nodes_ind)
            neg_nodes_ind = list(self.negs[node])
            node_neg = [node] * len(neg_nodes_ind)
            pos_score = self.cal_sim(self.embeddings[node_pos], self.embeddings[pos_nodes_ind])
            neg_score = self.cal_sim(self.embeddings[node_neg], self.embeddings[neg_nodes_ind])
            nodes_score.append(torch.max(torch.tensor(0.0), neg_score - pos_score + self.MARGIN).view(1, -1))
        loss = torch.mean(torch.cat(nodes_score, 0), 0)
        return loss

    def get_loss_margin_2(self):
        nodes_score = []
        nodes = list(self.adj_lists.keys())
        for node in nodes:
            pos_nodes_ind = list(self.pos[node])
            node_pos = [node] * len(pos_nodes_ind)
            neg_nodes_ind = list(self.negs[node])
            node_neg = [node] * len(neg_nodes_ind)

            pos_score = F.cosine_similarity(self.embeddings[node_pos], self.embeddings[pos_nodes_ind])
            pos_score = torch.log(torch.sigmoid(pos_score))

            neg_score = F.cosine_similarity(self.embeddings[node_neg], self.embeddings[neg_nodes_ind])
            neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))
        loss = torch.mean(torch.cat(nodes_score, 0), 0)
        return loss

    def get_loss_margin_3(self):
        _sum = torch.tensor(0)
        nodes = list(self.adj_lists.keys())
        _len = torch.tensor(len(nodes))
        for node in nodes:
            pos_nodes_ind = list(self.pos[node])
            neg_nodes_ind = list(self.negs[node])

            aff = torch.sum(self.embeddings[node] * self.embeddings[pos_nodes_ind], axis=1)
            neg_aff = torch.matmul(self.embeddings[node], torch.transpose(self.embeddings[neg_nodes_ind], 0, 1))

            true_xent = my_cross_entropy_with_logits(labels=torch.ones_like(aff), logits=aff)
            negative_xent = my_cross_entropy_with_logits(labels=torch.zeros_like(neg_aff), logits=neg_aff)
            _loss = torch.sum(true_xent) + self.Q * torch.sum(negative_xent)
            _sum = _sum+_loss
        loss = torch.div(_sum, _len)

        return loss

    def create_positives(self, method=2):
        if method is 1:
            self._run_random_walks()
        elif method is 2:
            self.get_random_walk()
        else:
            self.get_neighbours_sorted_by_weight()

        # print(self.pos)

    def create_neg(self):
        self.get_negtive_nodes()
        # print(self.negs)

    def _run_random_walks(self):
        nodes = list(self.adj_lists.keys())
        for node in nodes:
            if len(self.adj_lists[node]) == 0:
                print(str(node) + " is empty")
                continue
            for i in range(self.NUM_WALKS):
                curr_node = node
                for j in range(self.POS_WALK_LEN):
                    neighs = self.adj_lists[curr_node]
                    next_node = random.choice(list(neighs))
                    # Remove co-occurrences
                    if next_node != node and next_node in nodes:
                        if node in self.pos:
                            self.pos[node].add(next_node)
                        else:
                            self.pos[node] = set((next_node,))
                    curr_node = next_node

    def get_random_walk(self):
        nodes = list(self.adj_lists.keys())
        for node in nodes:
            for i in range(self.NUM_WALKS - 1):
                temp = list(self.graph.neighbors(node))
                # Remove co-occurrences
                temp = list(set(temp) - set([node]))
                if len(temp) == 0:
                    self.pos[node] = set((node,))
                    break
                random_node = random.choice(temp)
                if node in self.pos:
                    self.pos[node].add(random_node)
                else:
                    self.pos[node] = set((random_node,))
                node = random_node

    def get_neighbours_sorted_by_weight(self):
        nodes = list(self.adj_lists.keys())
        for node in nodes:
            temp = [x[0] for x in sorted([(i, j['weight']) for i, j in dict(self.graph[node]).items()], key=lambda x: x[-1], reverse=True)]

            # Remove co-occurrences
            temp = list(set(temp) - set([node]))
            if len(temp) == 0:
                 self.pos[node] = set((node,))
            elif len(temp) < self.NUM_WALKS - 1:
                self.pos[node] = set(temp)
            else:
                self.pos[node] = set(temp[0:self.NUM_WALKS])

    def get_negtive_nodes(self):
        num_neg = self.NUM_NEG
        nodes = list(self.adj_lists.keys())
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.NEG_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= set(self.adj_lists[outer])
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.adj_lists) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negs[node] = set(neg_samples)