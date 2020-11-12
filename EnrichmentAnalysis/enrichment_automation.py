"""
Script is used to compare biological process of selected geneset vs observed similar genesets.
"""

import csv
import sys
import ctypes as ct
import random

import math
from matplotlib.patches import Rectangle

from EnrichmentAnalysis.enrichment_utils import read_file, read_file_2, write_file

csv.field_size_limit(int(ct.c_ulong(-1).value//2))
import gseapy as gp
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as spy
import statsmodels.api as sm
import statistics
from scipy.stats import t


def res(x_one, y_one, top):
    array_one = []

    enr_x_one = None
    try:
        enr_x_one = gp.enrichr(gene_list=x_one, gene_sets=lib, organism='Human',
                               cutoff=0.05).results[['Term', 'P-value']].head(top).values.tolist()
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


lib = gp.get_library_name('Human')[54]
files = ["gae-hom-hom", "gae-hom-onto", "gae-onto-onto", "jcd-hom", "jcd-onto"]


def one_correlation():
    selected = read_file_2("../enrich_red/selected_genesets.csv")
    top_n = 20
    for i in files:
        similar = read_file_2("../enrich_red/"+i+".csv")

        array = []
        for x, y in zip(selected, similar):
            x_one = list(selected[x])
            y_one = list(similar[y])
            array.extend(res(x_one, y_one, top_n))

        x, y = zip(*array)

        fig, ax = plt.subplots()
        ax.scatter(x, y)

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
        # plt.text(i[2], i[3], 'R2 = %0.2f' % linreg.rvalue)
        # plt.text(i[2], i[4], 'Slope = %0.2f' % linreg.slope)
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        ax.legend([extra, extra], ('R2 = %0.2f' % linreg.rvalue, 'Slope = %0.2f' % linreg.slope))
        plt.title("Correlation plot for "+ i)
        plt.show()



def stat_correlation(perms, top_n):
    statistics = {"gae-hom-hom": list(), "gae-hom-onto": list(), "gae-onto-onto": list(),
                  "jcd-hom": list(), "jcd-onto": list()}

    for perm in range(perms):

        print("********** permutation no {} *********".format(perm))
        # random_elments = random.sample(list(read_file_2("ms-project/geneset_pairing.csv").keys()), 50)

        random_elments = random.sample(list(read_file_2("../enrich_red/selected_genesets.csv").keys()), 50)

        _selected = read_file_2("../enrich_red/selected_genesets.csv")

        selected = {ii: _selected[ii] for ii in random_elments}

        for i in files:
            print("file is {}".format(i))
            _similar = read_file_2("../enrich_red/" + i + ".csv")
            similar = {ii: _similar[ii] for ii in random_elments}

            array = []
            for x, y in zip(selected, similar):
                x_one = list(selected[x])
                y_one = list(similar[y])
                array.extend(res(x_one, y_one, top_n))

            x, y = zip(*array)

            linreg = spy.stats.linregress(x, y)

            statistics[i].append(linreg.rvalue)


    write_file("../perms/"+lib+".csv", statistics)


def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString


def qq_plot(data, title, filename):
    sm.qqplot(data, line='q')
    plt.xlabel("Theoritical Quantiles")
    plt.ylabel("Standard Residuals")
    plt.title("QQ--plot for "+ title)
    plt.savefig(filename, dpi=100)
    # plt.show()
    plt.clf()


def distribution():

    mthds = read_file_2("../perms2/GO_Biological_Process_2018.csv")

    # Using 95% confidence interval
    # (1-0.95)/2
    t_score = abs(t.ppf(0.025, 23))
    alpha = 1 - 0.95

    excel_rows = [
        ['Method', 'PT. Est', 'lower CI', 'upper CI']]

    for i in mthds:

        mean = statistics.mean(mthds[i])
        std = statistics.stdev(mthds[i])
        sqtr_nu = math.sqrt(len(mthds[i]))
        # p_hat and q_hat set to conservative since we have no previous data #0.5 for each
        # Since its probability I clip to 0

        x = pd.Series(mthds[i])


        # if i == 'jcd-hom-jcd-onto':
        #     mtd = replacenth(i, "-", " vs ", 2)
        # else:
        #     mtd = replacenth(i, "-", " vs ", 3)
        mtd = i
        lower_ci = max(mean - t_score * std / sqtr_nu, 0)
        upper_ci = mean + t_score * std/sqtr_nu

        qq_plot(x, mtd, "../qqplots/qq-"+i+".png")

        excel_rows.append([mtd, round(mean, 3), round(lower_ci, 3), round(upper_ci, 3)])

    df = pd.DataFrame.from_records(excel_rows[1:], columns=excel_rows[0])

    print(df)
    print(df.to_latex(index=True))



# stat_correlation(10, 2)
distribution()