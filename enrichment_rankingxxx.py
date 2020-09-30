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


# Annotation for scores
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
files = ["gae-hom-hom", "gae-hom-onto", "gae-onto-onto", "gcn-hom-hom",
         "gcn-hom-onto", "gcn-onto-onto", "jcd-hom-hom", "jcd-onto-onto"]


writer = pd.ExcelWriter('annot/data.xlsx')
for i in files:
    file_name = "enrich/"+i+".csv"
    file = read_file(file_name)
    print("file "+str(i))
    temp = pd.DataFrame()
    for j in file:
        gene_list = list(file[j])

        try:
            enr_x_one = gp.enrichr(gene_list=gene_list, gene_sets=lib, organism='Human',
                                   cutoff=0.05).results[['Gene_set', 'Term', 'P-value',
                                                        'Genes']].head(5)
            # enr_x_one['genesetID'] = j
            # enr_x_one['nearest-4-geneset'] = j
            # enr_x_one['neigbours'] =
            # temp = temp.append(enr_x_one)
            #print(temp)
        except Exception:
            pass

    temp.to_excel(writer, sheet_name=i)


writer.save()




