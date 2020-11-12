"""
Script is used to generate the enrichment results , an excel file with the following row:

'genesetID', 'Gene Set Name', 'Gene Set Description',
'Neighbour 1 Name', 'Neighbour 1 Desc'
'Neighbour 2 Name', 'Neighbour 2 Desc',
'Neighbour 3 Name', 'Neighbour 3 Desc'
'Neighbour 4 Name', 'Neighbour 4 Desc'

"""

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

from EnrichmentAnalysis.enrichment_utils import read_file, read_file_2, write_file


def reduce_genesets():
    sample = read_file("enrich_red/gae-hom-hom.csv")
    red = read_file("enrich_red/selected_genesets.csv")
    temp = {}
    for i in red:
        if i in sample:
            temp[i] = red[i]
    write_file("enrich_red/selected_genesets.csv", temp)

# reduce_genesets()


lib = gp.get_library_name('Human')[53]
files = [("gae-hom-hom", 1, 1), ("gae-hom-onto", 1, 2), ("gae-onto-onto", 2, 3),
         ("jcd-hom-hom", 1, 4), ("jcd-onto-onto", 2, 5)]
data_desc = read_file_2("data\ms-project\data-description.csv")



for i in files:
    file_name = "enrich_red/"+i[0]+".csv"
    file = read_file(file_name)
    # if i[1] == 1:
    #     neigh = read_file_2("data/ms-project/neig_len_hom.csv")
    # else:
    #     neigh = read_file_2("data/ms-project/neig_len_onto.csv")

    if i[2] == 1:
        rank = read_file_2("ranking_results/ -- GAE -- Homology -- Homology.csv")
    elif i[2] == 2:
        rank = read_file_2("ranking_results/ -- GAE -- Homology -- Ontology.csv")
    elif i[2] == 3:
        rank = read_file_2("ranking_results/ -- GAE -- Ontology -- Ontology.csv")
    elif i[2] == 4:
        rank = read_file_2("ranking_results/--JCD--Homology.csv")
    elif i[2] == 5:
        rank = read_file_2("ranking_results/--JCD--Ontology.csv")

    temp = pd.DataFrame()
    for j in file:
        gene_list = list(file[j])

        enr_x_one = None
        try:
            enr_x_one = gp.enrichr(gene_list=gene_list, gene_sets=lib, organism='Human',
                                   cutoff=0.05).results[['Term', 'P-value',
                                                        'Genes']].head(5)
        except:
            pass

        if enr_x_one is not None:
            enr_x_one['genesetID'] = j
            enr_x_one['Gene Set Name'] = data_desc[j][0]
            enr_x_one['Gene Set Description'] = data_desc[j][1]

            temple = rank[j]

            n1, n1d, n2, n2d, n3, n3d, n4, n4d = "", "", "", "", "", "", "", ""

            n1 = data_desc[str(temple[0])][0]
            n1d = data_desc[str(temple[0])][1]
            enr_x_one['Neighbour 1 Name'] = n1
            enr_x_one['Neighbour 1 Desc'] = n1d

            n2 = data_desc[str(temple[1])][0]
            n2d = data_desc[str(temple[1])][1]
            enr_x_one['Neighbour 2 Name'] = n2
            enr_x_one['Neighbour 2 Desc'] = n2d

            n3 = data_desc[str(temple[2])][0]
            n3d = data_desc[str(temple[2])][1]
            enr_x_one['Neighbour 3 Name'] = n3
            enr_x_one['Neighbour 3 Desc'] = n3d

            n4 = data_desc[str(temple[3])][0]
            n4d = data_desc[str(temple[3])][1]
            enr_x_one['Neighbour 4 Name'] = n4
            enr_x_one['Neighbour 4 Desc'] = n4d

            # enr_x_one['neigbours'] = neigh[j]
            # enr_x_one['sufficient_neighbours'] = True if int(neigh[j]) > 4 else False
            temp = temp.append(enr_x_one)

    writer = pd.ExcelWriter('annot/'+i[0]+'.xlsx')
    temp.to_excel(writer, sheet_name="sheet1")
    writer.save()







