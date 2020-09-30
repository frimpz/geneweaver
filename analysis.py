import csv
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def read_file(file_name):
    """
        Method reads a dictionary type object from file
        :param file_name: filename
        : return : dictionary
        """
    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        mydict = dict(reader)
        mydict = {int(u): eval(v) for u, v in mydict.items()}
    return mydict


def read_file2(file_name):
    with open(file_name) as f:
        lis = [line.split() for line in f][1:]
        dic_set = {}
        for i in lis:
            key = int(i[0])
            value = int(i[1])
            if key in dic_set:
                dic_set[key].append(value)
            else:
                dic_set[key] = [value]
        return dic_set


def jac_sim(dic, u, v):
    """
        Method calculates jaccard similarity between two nodes
        :param dic:
        :param u: node 1
        :param v: node 2
        """
    try:
        return len(frozenset.intersection(dic[u], dic[v]))/len(frozenset.union(dic[u], dic[v]))
    except TypeError:
        return 0
    except ZeroDivisionError:
        return 0


def cal_jaccard(gs_1, gs_2):
    dic = read_file('data/gene/node_graph_2.csv')
    return jac_sim(dic, gs_1, gs_2)


def cal_jaccards(gs_1, top_k=5):
    dic = read_file('data/gene/node_graph_2.csv')
    scores = []
    for i in dic.keys():
        scores.append((i, cal_jaccard(gs_1, i)))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[: top_k]
    return scores

# https://ragrawal.wordpress.com/2013/01/18/comparing-ranked-list/
def score(l1, l2, p=0.98):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


# print(score([1, 2, 3, 4, 5], [2, 1, 3, 5, 4], p=0.98))


# get random genseets
samples = random.sample(range(1240), 28)
res = read_file("data/gene/geneset_pairing.csv")
res = dict((v, k) for k, v in res.items())
samples = [res[i] for i in samples]

with open('results/samples.pkl', 'wb') as f:
    pickle.dump(samples, f)

with open('results/samples.pkl', 'rb') as fp:
    samples = pickle.load(fp)

samples.extend([75652, 1047, 128199, 271963, 216492, 75651, 75646, 75649, 75644, 75633, 248620, 75612,
                75599, 271937, 213752, 75550, 271938, 217567, 216148, 1047, 865,  271956])
random.shuffle(samples)

gae = read_file2('results/ranking_gae.csv')
gcn = read_file2('results/ranking_gcn.csv')
w_adj_dic = read_file("data/gene/adj_2.csv")

for i in samples:
    temp = w_adj_dic[i]
    temp = sorted([(i, temp[i]['weight']) for i in temp.keys()], key=lambda xx: xx[1], reverse=True)[0:20]
    temp = [i[0] for i in temp]

ps = [0.5, 0.8, 0.9, 0.95]


for p in ps:
    z = []
    for i in samples:
        temp = w_adj_dic[i]
        temp = sorted([(i, temp[i]['weight']) for i in temp.keys()], key=lambda xx: xx[1], reverse=True)[0:20]
        temp = [i[0] for i in temp]
        z.append((i, score(gae[i], temp, p=p), score(gcn[i], temp, p=p)))
        print((i, score(gae[i], temp, p=p), score(gcn[i], temp, p=p)))

    y1 = []
    y2 = []
    for i in z:
        y1.append(i[1])
        y2.append(i[2])
        print(i, y1, y2)

    x = range(len(y1))
    x_min = min(x) - 0.1
    x_max = max(x) + 0.2

    y_min = min(y1 + y2) - 0.1
    y_max = max(y1 + y2) + 0.2

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot(x, y1, 'r', label='GAE', marker=(8, 2, 0), ls='--')
    plt.plot(x, y2, 'b', label='GCN', marker="+", ls=':')
    plt.xlabel('Geneset')
    plt.ylabel('RBO to Jaccard')
    plt.title("RBO PLOT p= " + str(p))
    plt.legend(loc="upper right")
    plt.show()


    plt.clf()

    x_min = min(y1) - 0.1
    x_max = max(y1) + 0.2

    y_min = min(y2) - 0.1
    y_max = max(y2) + 0.2

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.scatter(y1, y2, color='black', label='')
    linreg = stats.linregress(y1, y2)
    temp = [linreg.intercept + linreg.slope.item() * i for i in y1]
    plt.plot(y1, temp, 'r', label='')

    plt.text(-0.05, 1.0, 'R2 = %0.2f' % linreg.rvalue)
    plt.text(-0.05, 0.96, 'Slope = %0.2f' % linreg.slope)

    plt.xlabel('GAE')
    plt.ylabel('GCN')
    plt.title("RBO Correlation PLOT p= " + str(p))
    plt.legend(loc="upper right")
    plt.show()

    # x = read_file2('results/ranking_gae.csv')
    # for i in x:
    #     print(i, x[i])



    # x = read_file2('results/ranking_gcn.csv')
    # for i in x:
    #     print(i, x[i])


    # for i in samples:
    #     # x = cal_jaccards(i, top_k=10)
    #     print(i)


    # print(cal_jaccard(271966, 271953))
    # print(cal_jaccards(834, top_k=10))
