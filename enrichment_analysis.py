import csv
import ctypes as ct
import sys

csv.field_size_limit(int(ct.c_ulong(-1).value//2))
import gseapy as gp
import pandas as pd
from utils import  write_file
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plots import heat_map


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
        my_dict = {u: eval(v) for u, v in my_dict.items()}
        return my_dict

# connection graph and correlation plot
def save_enrichment(x):
     lib = gp.get_library_name('Human')

     with open('gensets.txt', 'w') as f:
          for item in range(len(lib)):
               f.write("%s %s\n" % (item, lib[item]))
     # lib = lib[49: 54]
     lib = lib[53]

     files = [(1, x+"/gcn-hom-hom.csv"), (2, x+"/gcn-hom-onto.csv"),
              (3, x+"/gcn-onto-onto.csv"), (4, x+"/gae-hom-hom.csv"),
              (5, x+"/gae-hom-onto.csv"), (6, x+"/gae-onto-onto.csv")]

     df = pd.DataFrame()
     writer = pd.ExcelWriter('enrich-cluster/full-results.xlsx')
     for key, file in files:
          print(file)
          cluster_data = read_file(file)
          for i in cluster_data:
               try:
                    enr = gp.enrichr(gene_list=list(cluster_data[i][2]), gene_sets=lib, organism='Human', cutoff=0.05).results
               except:
                    pass
               enr['model'] = key
               enr['cluster'] = i
               df = df.append(enr)

     df = df[(df['P-value'] < 0.05)]
     df.to_excel(writer, sheet_name="sheet1")
     writer.save()



def save_enrichment_set():

     lib = gp.get_library_name('Human')
     lib = lib[53]

     files = [("gcn-hom-hom", "enrich/gcn-hom-hom.csv"),
              ("gcn-hom-onto", "enrich/gcn-hom-onto.csv"),
              ("gcn-onto-onto", "enrich/gcn-onto-onto.csv"),
              ("gae-hom-hom", "enrich/gae-hom-hom.csv"),
              ("gae-hom-onto", "enrich/gae-hom-onto.csv"),
              ("gae-onto-onto", "enrich/gae-onto-onto.csv")]

     enrich_set = {}
     for key, file in files:
          print(file)
          cluster_data = read_file(file)
          for i in cluster_data:
              print(len(cluster_data[i][2]))
              try:
                  enr = gp.enrichr(gene_list=list(cluster_data[i][2])[:1000], gene_sets=lib, organism='Human', cutoff=0.05).results
                  name = key + "-" + str(i)
                  term = enr['Term'].to_list()
                  enrich_set[name] = term
                  # print(i)
                  print(enr)
              except:
                   pass


     write_file("enrich-cluster/full_result_dic.csv", enrich_set)
save_enrichment_set()


x = read_file("enrich-cluster/full_result_dic.csv")

l1 = {}
for i in x.keys():
     l2 = []
     for j in x.keys():
          l2.append(jac_sim(x[i], x[j]))
     l1[i] = l2


df = pd.DataFrame.from_dict(l1, orient='index').transpose()
col = {}
col_name = list(df.columns.values)
for i in range(len(col_name)):
    col[col_name[i]] = i
print(col)
df = df.rename(columns=col)
print(df)

# df = df[df > 0.5]

print(df.to_latex())

fig, ax = plt.subplots(figsize=(10, 6))
hm = sns.heatmap(round(df, 2))
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
hm.set_title('Correlation plot for enrichment', fontsize=14)
plt.show()

