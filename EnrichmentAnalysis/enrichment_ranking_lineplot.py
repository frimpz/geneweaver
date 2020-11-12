"""
Script is used to create the bar plot between pairwise biological processes
of each different method.

'gae-hom-hom' vs 'jcd-hom'
'gae-hom-onto' vs 'jcd-hom'
'gae-onto-onto' vs 'jcd-onto'
'gae-hom-onto' vs 'jcd-onto'

"""

import csv
import sys
import ctypes as ct
csv.field_size_limit(int(ct.c_ulong(-1).value//2))
import gseapy as gp
import re
import statistics as stat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from EnrichmentAnalysis.enrichment_utils import read_file, read_file_2, write_file

lib = gp.get_library_name('Human')[53]
files = ["gae-hom-hom", "gae-hom-onto", "gae-onto-onto", "jcd-hom", "jcd-onto"]


def save_top4_csvs():
    for i in files:
        file_name = "enrich/" + i + ".csv"
        file = read_file(file_name)

        temp = {}
        for j in file:
            gene_list = list(file[j][2])

            enr_x_one = None
            try:
                enr_x_one = gp.enrichr(gene_list=gene_list, gene_sets=lib, organism='Human',
                                       cutoff=0.05).results.head(10)['Term'].tolist()
                temp[j] = set(enr_x_one)
            except:
                pass
        write_file("top_4_bio_process/" + i + ".csv", temp)


# save_top4_csvs()

keys = [271955, 218445, 34466, 127372, 614, 219270, 1051, 246704, 34620, 75542, 221932,
        34582, 644, 127417, 617, 246616, 218722, 35882, 137565, 1214, 216888, 46984, 553,
        839, 14904, 35877, 135191, 34438, 75646, 215984, 921, 75761, 137384, 135194, 271641,
        866, 75683, 36285, 218726, 34682, 1766, 216881, 216895, 34302, 845, 34549, 611, 34518,
        225899, 271950, 1072, 75604, 34560, 219217, 225900, 218721, 36299, 36287, 34513, 75623,
        34904, 34608, 865, 566, 14892, 34625, 137861, 222630, 271725, 34552, 216885, 810, 37188,
        552, 271940, 35883, 573, 75657, 355584, 34970, 271727, 34439, 34883, 34786, 271884, 219275,
        608, 75644, 271096, 34520, 218958, 1267, 75654, 34314, 246395, 273371, 36320, 267511, 26324, 802]

def line_plots():
    pairs = [('gae-hom-hom', 'jcd-hom'), ('gae-hom-onto', 'jcd-hom'),
             ('gae-onto-onto', 'jcd-onto'), ('gae-hom-onto', 'jcd-onto')]


    dic = {}
    for i in pairs:

        file_one = read_file("top_4_bio_process/" + i[0] + ".csv")
        file_two = read_file("top_4_bio_process/" + i[1] + ".csv")

        name = i[0]+"-"+i[1]

        dic[name] = {0:0, 5:0, 10:0, 15:0, 20:0}

        for j in keys:
            x = file_one[str(j)] if str(j) in file_one else set()
            y = file_two[str(j)] if str(j) in file_two else set()
            intersect = len(x & y)

            if intersect == 0:
                dic[name][0] = dic[name][0] + 1
            elif intersect < 5:
                dic[name][5] = dic[name][5] + 1
            elif intersect < 10:
                dic[name][10] = dic[name][10] + 1
            elif intersect < 15:
                dic[name][15] = dic[name][15] + 1
            else:
                dic[name][20] = dic[name][20] + 1

    # normalise
    # for i in dic:
    #     mean = 20
    #     std = stat.stdev(dic[i].values())
    #
    #     for j in dic[i]:
    #         dic[i][j] = dic[i][j]/100

    pd.DataFrame(dic).plot(kind='bar')
    plt.xlabel("# of nearest x")
    plt.ylabel("# of Occurrence")
    plt.title("# of Nearest-x-genesets in common between selected graphs")
    plt.show()


    # write_file("top_4_bio_process/result.csv", dic)

line_plots()


