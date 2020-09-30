import csv
import sys
import ctypes as ct
csv.field_size_limit(int(ct.c_ulong(-1).value//2))
import gseapy as gp
import re
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

# "gae-hom-hom", "gae-hom-onto"), ("gae-hom-hom", "gae-onto-onto"), ("gae-hom-hom", "gae-hom-hom"),
#          ("gae-hom-hom", "gcn-hom-hom"), ("gae-hom-hom", "gcn-hom-onto"),
#          ("gae-hom-hom", "gcn-onto-onto"), ("gae-hom-hom", "jcd-hom-hom"),
#          ("gae-hom-hom", "jcd-onto-onto"), ("gae-hom-onto", "jcd-onto-onto"),

lib = gp.get_library_name('Human')[53]
files = [
         ("gae-hom-onto", "gae-onto-onto"), ("gae-hom-onto", "gcn-hom-hom"),
         ("gae-hom-onto", "gcn-hom-onto"), ("gae-hom-onto", "gcn-onto-onto"),
         ("gae-hom-onto", "jcd-hom-hom"), ("gae-onto-onto", "jcd-onto-onto"),
         ("gae-onto-onto", "gcn-hom-hom"), ("gae-onto-onto", "gcn-hom-onto"),
         ("gae-onto-onto", "gcn-onto-onto"), ("gae-onto-onto", "jcd-hom-hom"),
         ("gcn-hom-hom", "jcd-onto-onto"), ("gcn-hom-hom", "gcn-hom-onto"),
         ("gcn-hom-hom", "gcn-onto-onto"), ("gcn-hom-hom", "jcd-hom-hom"),
         ("gcn-hom-onto", "jcd-onto-onto"), ("gcn-hom-onto", "gcn-onto-onto"),
         ("gcn-onto-onto", "jcd-onto-onto"), ("gcn-onto-onto", "jcd-hom-hom"),
         ("gcn-hom-onto", "jcd-hom-hom"), ("jcd-hom-hom", "jcd-onto-onto")]



for i in files:
    file_one = "enrich/"+i[0]+".csv"
    file_two = "enrich/"+i[1]+".csv"
    one = read_file(file_one)
    two = read_file(file_two)

    array_one = []
    array_two = []
    counter = 0
    for x, y in zip(one, two):
        x_one = list(one[x])
        y_one = list(two[y])
        # ['Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
        #        'Old P-value', 'Old Adjusted P-value', 'Odds Ratio', 'Combined Score',
        #        'Genes']

        try:
            enr_x_one = gp.enrichr(gene_list=x_one, gene_sets=lib, organism='Human',
                                   cutoff=0.05).results[['Term', 'P-value',
                                                         'Combined Score']].iloc[0].values.tolist()

            enr_y_one = gp.enrichr(gene_list=y_one, gene_sets=lib, organism='Human',
                                   cutoff=1).results[['Term', 'P-value',
                                                      'Combined Score']]
            try:
                enr_y_one = enr_y_one.loc[enr_y_one['Term'] == enr_x_one[0]].iloc[0].values.tolist()
                array_one.append((enr_x_one[0], get_code_name(enr_x_one[0])[0], enr_x_one[1], enr_y_one[1]))
            except IndexError:
                array_one.append((enr_x_one[0], get_code_name(enr_x_one[0])[0], enr_x_one[1], 1.0))
        except Exception:
            pass

        try:
            enr_x_one = gp.enrichr(gene_list=y_one, gene_sets=lib, organism='Human',
                                   cutoff=0.05).results[['Term', 'P-value',
                                                         'Combined Score']].iloc[0].values.tolist()

            enr_y_one = gp.enrichr(gene_list=x_one, gene_sets=lib, organism='Human',
                                   cutoff=1).results[['Term', 'P-value',
                                                      'Combined Score']]
            try:
                enr_y_one = enr_y_one.loc[enr_y_one['Term'] == enr_x_one[0]].iloc[0].values.tolist()
                array_two.append((enr_x_one[0], get_code_name(enr_x_one[0])[0], enr_x_one[1], enr_y_one[1]))
            except IndexError:
                array_two.append((enr_x_one[0], get_code_name(enr_x_one[0])[0], enr_x_one[1], 1.0))
        except Exception:
            pass

    x_1, y_1 = zip(*[(i[2], i[3]) for i in array_one])
    y_2, x_2 = zip(*[(i[2], i[3]) for i in array_two])

    plt.plot(x_1, y_1, 'g*', x_2, y_2, 'ro')
    plt.ylabel(i[0])
    plt.xlabel(i[1])
    plt.legend([i[0], i[1]], loc="upper right")
    plt.title(i[0] + str(" vs ") + i[1])
    plt.show()



