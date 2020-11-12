"""
Script is used to create the scatter plot between pairwise biological processes
of each different method.

GAE-Hom-Only vs GAE-Hom-Onto
GAE-Hom-Only vs GAE-Onto-Only
GAE-Hom-Only vs Jaccard-Homology
GAE-Hom-Only vs Jaccard-Ontology
GAE-Hom-Onto vs GAE-Onto-Only
GAE-Hom-Onto vs Jaccard-Homology
GAE-Hom-Onto vs Jaccard-Ontology
GAE-Onto-Only vs Jaccard-Homology
GAE-Onto-Only vs Jaccard-Ontology
Jaccard-Homology vs Jaccard-Ontology
"""

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
import random
import seaborn as sns
import matplotlib.pyplot as plt

from EnrichmentAnalysis.enrichment_utils import read_file, read_file_2, write_file

lib = gp.get_library_name('Human')[54]


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


files = [("gae-hom-hom", "gae-hom-onto"),
         ("gae-hom-hom", "gae-onto-onto"),
         ("gae-hom-hom", "jcd-hom"),
         ("gae-hom-hom", "jcd-onto"),

         ("gae-hom-onto", "gae-onto-onto"),
         ("gae-hom-onto", "jcd-hom"),
         ("gae-hom-onto", "jcd-onto"),

         ("gae-onto-onto", "jcd-hom"),
         ("gae-onto-onto", "jcd-onto"),

         ("jcd-hom", "jcd-onto")]


def scatter_plots():
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


def scatter_stats(perms):

    statistics = {"gae-hom-hom-gae-hom-onto": list(),
                  "gae-hom-hom-gae-onto-onto": list(),
                  "gae-onto-onto-jcd-onto": list(),
                  "gae-hom-hom-jcd-hom": list(),
                  "gae-hom-hom-jcd-onto": list(),
                  "gae-hom-onto-gae-onto-onto": list(),
                  "gae-hom-onto-jcd-hom": list(),
                  "gae-hom-onto-jcd-onto": list(),
                  "gae-onto-onto-jcd-hom": list(),
                  "jcd-hom-jcd-onto": list(),
                  }

    for perm in range(perms):

        print("********** permutation no {} *********".format(perm))

        random_elments = random.sample(list(read_file_2("../enrich_red/selected_genesets.csv").keys()), 50)


        for i in files:
            print("file is {}".format(i))
            file_one = "../enrich_red/" + i[0] + ".csv"
            file_two = "../enrich_red/" + i[1] + ".csv"

            _one = read_file_2(file_one)
            _two = read_file_2(file_two)

            one = {ii: _one[ii] for ii in random_elments}
            two = {ii: _two[ii] for ii in random_elments}

            array = []
            for x, y in zip(one, two):
                x_one = list(one[x])
                y_one = list(two[y])

                array.extend(res(x_one, y_one))
                array.extend(res(y_one, x_one))

            x, y = zip(*array)

            linreg = spy.stats.linregress(x, y)

            statistics[i[0]+"-"+i[1]].append(linreg.rvalue)

    write_file("../perms2/"+lib+".csv", statistics)


scatter_stats(10)




