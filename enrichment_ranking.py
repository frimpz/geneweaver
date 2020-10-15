import csv
import sys
import ctypes as ct
csv.field_size_limit(int(ct.c_ulong(-1).value//2))
import gseapy as gp
import scipy as spy
import re
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Ranking scatter plot
def jac_sim(u, v):
    """
        Method calculates jaccard similarity between two nodes
        :param p1: dictionary
        :param p2: node 1
        :param p2: node 2
        """
    try:
        return round(len(set.intersection(u, v))/len(set.union(u, v)), 2)
    except TypeError:
        return 0
    except ZeroDivisionError:
        return 0


def read_file(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        my_dict = dict(reader)
        my_dict = {u: eval(v)[2] for u, v in my_dict.items()}
        return my_dict


# get codename in parenthesis
def get_code_name(s):
    res = [i.strip("()") for i in re.findall(r'\(.*?\)', s)]
    return res


lib = gp.get_library_name('Human')[53]


def res(x_one, y_one):
    array_one = []
    enr_x_one = None
    try:
        enr_x_one = gp.enrichr(gene_list=x_one, gene_sets=lib, organism='Human',
                               cutoff=0.05).results[['Term', 'P-value']].head(10).values.tolist()
    except Exception:
        pass

    if enr_x_one is not None and len(enr_x_one) > 0:
        enr_y_one = None

        try:
            enr_y_one = gp.enrichr(gene_list=y_one, gene_sets=lib, organism='Human',
                                   cutoff=1.0).results[['Term', 'P-value']]
        except Exception:
            pass

        if enr_y_one is not None:
            for term in enr_x_one:
                pair = enr_y_one.loc[enr_y_one['Term'] == term[0]]
                if pair is not None and pair.shape[0] > 0:
                    pair = pair.iloc[0].values.tolist()
                    array_one.append((term[1], pair[1]))

    return array_one


files = [("gae-hom-hom", "gae-hom-onto", 0.5, 1.15, 1.10),
         ("gae-hom-hom", "gae-onto-onto", 0.5, 0.9, 0.8),
         ("gae-hom-hom", "jcd-hom-hom", 0.10, 0.9, 0.8),
         ("gae-hom-hom", "jcd-onto-onto", 0.125, 1.0, 0.9),

         ("gae-hom-onto", "gae-onto-onto", 0.5, 0.9, 0.8),
         ("gae-hom-onto", "jcd-hom-hom", 0.10, 0.9, 0.8),
         ("gae-hom-onto", "jcd-onto-onto", 0.2, 0.9, 0.8),

         ("gae-onto-onto", "jcd-hom-hom", 0.5, 1.15, 1.10),
         ("gae-onto-onto", "jcd-onto-onto", 0.1, 1.15, 1.10),

         ("jcd-hom-hom", "jcd-onto-onto", 0.11, 0.9, 0.8)]

files = [("gae-hom-onto", "jcd-onto-onto", 0.2, 0.9, 0.8),]

for i in files:
    file_one = "enrich_red/"+i[0]+".csv"
    file_two = "enrich_red/"+i[1]+".csv"
    one = read_file(file_one)
    two = read_file(file_two)

    array = []
    for x, y in zip(one, two):
        x_one = list(one[x])
        y_one = list(two[y])

        array.extend(res(x_one, y_one))
        array.extend(res(y_one, x_one))

    x, y = zip(*array)

    plt.scatter(x, y)
    linreg = spy.stats.linregress(x, y)
    temp = [linreg.intercept + linreg.slope.item() * k for k in x]
    plt.plot(x, temp, 'r')

    x_min = min(x)
    x_max = max(x) + statistics.stdev(x)

    y_min = min(y)
    y_max = max(y) + statistics.stdev(y)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.ylabel("Corresponding P-value")
    plt.xlabel("Actual P-value")
    plt.text(i[2], i[3], 'R2 = %0.2f' % linreg.rvalue)
    plt.text(i[2], i[4], 'Slope = %0.2f' % linreg.slope)
    plt.title(i[0]+" and "+i[1])
    plt.show()




