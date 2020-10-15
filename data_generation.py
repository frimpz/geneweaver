# This file is used to create the data for the node clustering.
# G(V,E) -- undirected graph
# from itertools import chain
import itertools
# todo: continue documentation
import sys
from operator import getitem

import networkx as nx
import scipy
from psycopg2.pool import ThreadedConnectionPool
import csv
import pandas as pd
import numpy as np

# This class is used to create a connection to the DBase
class GeneWeaverThreadedConnectionPool(ThreadedConnectionPool):
    def __init__(self, minconn, maxconn, *args, **kwargs):
        ThreadedConnectionPool.__init__(self, minconn, maxconn, *args, **kwargs)
    def _connect(self, key=None):
        conn = super(GeneWeaverThreadedConnectionPool, self)._connect(key)
        conn.set_client_encoding('UTF-8')
        cursor = conn.cursor()
        cursor.execute('SET search_path TO production, extsrc, odestatic;')
        conn.commit()
        return conn

# Instantiate the database connection. class
pool = GeneWeaverThreadedConnectionPool(5, 20,
                                        database='geneweaver',
                                        user='gene',
                                        password='gene',
                                        host='localhost',
                                        port=5432)

# Create pooled cursor class
class PooledCursor(object):
    def __init__(self, conn_pool=pool, rollback_on_exception=False):
        self.conn_pool = conn_pool
        self.rollback_on_exception = rollback_on_exception
        self.connection = None

    def __enter__(self):
        self.connection = self.conn_pool.getconn()
        return self.connection.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection is not None:
            if self.rollback_on_exception and exc_type is not None:
                self.connection.rollback()

            self.conn_pool.putconn(self.connection)


def write_file(file_name, dic):
    """
    Method writes a dictionary type object to file
    :param p1: filename
    :param p2: dictionary object
    : return : None
    """
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=':')
        for key, value in dic.items():
            writer.writerow([key, value])


def read_file(file_name):
    """
        Method reads a dictionary type object from file
        :param p1: filename
        : return : dictionary
        """
    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        mydict = dict(reader)
        mydict = {int(u): eval(v) for u,v in mydict.items()}
    return mydict


def get_gs_ids():
    """
    Method returns the genesets in the graph
    You can ask Dr. Baker for curation group, he stated that it will reduce the total number of nodes
    : return : cursor of geneset IDS
    """
    with PooledCursor() as cursor:
        cursor.execute(
            '''
            SELECT gs.gs_id, gs.gs_name, gs.gs_description
            FROM production.geneset gs 
            WHERE gs.cur_id=3
            AND gs.gs_status 
            NOT LIKE 'de%' order by gs.gs_id;
            '''
        )
    return cursor

res = get_gs_ids()
pi = [i for i in res.fetchall()]
dic = {}
for i in pi:
    dic[i[0]] = i[1:]

write_file('ms-project/data-description.csv', dic)


def get_homology(geneset):
    """
    Method returns the homology for each geneset selected
    :param p1 : geneset Id
    : return : cursor of geneset IDS
    """
    with PooledCursor() as cursor:
        cursor.execute(
            '''
            SELECT hom_id
            FROM extsrc.geneset_value AS gs 
            JOIN extsrc.homology as hm
            ON gs.ode_gene_id = hm.ode_gene_id
            WHERE gs.gs_id = %s;
            ''',
            (geneset,)
        )
    return cursor

def get_ontology(geneset):
    """
    Method returns the ontology for each geneset selected
    :param p1 : geneset Id
    : return : cursor of geneset IDS
    """
    with PooledCursor() as cursor:
        cursor.execute(
            '''
            SELECT ont_id 
            FROM extsrc.geneset_ontology 
            WHERE gs_id = %s;
            ''',
            (geneset,)
        )
    return cursor

def get_genes(homology):
    """
    Method returns the genes and species based on homology
    :param p1 : Homology Id
    : return : cursor of gene IDS and species
    """
    with PooledCursor() as cursor:
        cursor.execute(
            '''
            select g.ode_gene_id, g.ode_ref_id, g.sp_id 
            from gene g inner join homology h 
            on g.ode_gene_id=h.ode_gene_id  
            where g.gdb_id=7 
            and h.sp_id=2 
            and h.hom_id= %s;
            ''',
            (homology,)
        )
    return cursor


def to_onr_set(x):
    return {i[0] for i in x}


def cal_neighbours():
    hom = read_file("ms-project/grp_gene.csv")
    data = {}
    for i in hom:
        data[i] = len([ii for ii in hom[i] if hom[i][ii]['weight']>0])
    write_file('ms-project/neig_len_hom.csv', data)

    onto = read_file("ms-project/grp_ontology.csv")
    data = {}
    for i in onto:
        data[i] = len([ii for ii in onto[i] if onto[i][ii]['weight']>0])
    write_file('ms-project/neig_len_onto.csv', data)
cal_neighbours()


def pair_hom_genes(filename):
    clusters = {}
    with open('csvs/kmeans_gcn_hom_hom.csv') as f:
        next(f)
        lines = f.read().splitlines()
        for line in lines:
            line = line.split(",")
            key = line[2]
            value = line[1]
            if key in clusters:
                clusters[key].add(value)
            else:
                clusters[key]= set(value)

    assoc_homs = {}
    for cluster in clusters.keys():
        print(cluster)
        for geneset in clusters[cluster]:
            if cluster in assoc_homs:
                assoc_homs[cluster] |= to_onr_set(get_homology(geneset))
            else:
                assoc_homs[cluster] = to_onr_set(get_homology(geneset))

    enrich = {}
    for geneset in assoc_homs.keys():
        # Iterate over homology
        _homs = set()
        _genes = set()
        _refs = set()
        print("geneset :"+str(geneset))
        for homology in assoc_homs[geneset]:
            res = get_genes(homology)
            for i in res.fetchall():
                _genes.add(i[0])
                _refs.add(i[1])
            _homs.add(homology)
        enrich[geneset] = [_homs, _genes, _refs]

    for i in enrich:
        print(i, len(enrich[i][1]), enrich[i])

    # Just for future use
    write_file('enrich-cluster/' + filename + '.csv', enrich)
# pair_hom_genes("gcn-hom-hom")


# get ode_refs from similar genesets
def geneset_odes(filename):
    ranked_genesets = read_file("ranking_results/ -- GAE -- Ontology -- Ontology.csv")
    dic = [271955, 218445, 34466, 127372, 614, 219270, 1051, 246704, 34620, 75542, 221932, 34582, 644, 127417, 617, 246616, 218722, 35882, 137565, 1214, 216888, 46984, 553, 839, 14904, 35877, 135191, 34438, 75646, 215984, 921, 75761, 137384, 135194, 271641, 866, 75683, 36285, 218726, 34682, 1766, 216881, 216895, 34302, 845, 34549, 611, 34518, 225899, 271950, 1072, 75604, 34560, 219217, 225900, 218721, 36299, 36287, 34513, 75623, 34904, 34608, 865, 566, 14892, 34625, 137861, 222630, 271725, 34552, 216885, 810, 37188, 552, 271940, 35883, 573, 75657, 355584, 34970, 271727, 34439, 34883, 34786, 271884, 219275, 608, 75644, 271096, 34520, 218958, 1267, 75654, 34314, 246395, 273371, 36320, 267511, 26324, 802]

    ranked_genesets = {i: ranked_genesets[i] for i in dic}

    assoc_homs = {}
    for rank in ranked_genesets:
        for geneset in ranked_genesets[rank]:
            if rank == geneset:
                pass
            if rank in assoc_homs:
                assoc_homs[rank] |= to_onr_set(get_homology(geneset))
            else:
                assoc_homs[rank] = to_onr_set(get_homology(geneset))

    print("loop1")

    enrich = {}
    for geneset in assoc_homs.keys():
        print(geneset)
        # Iterate over homology
        _homs = set()
        _genes = set()
        _refs = set()
        for homology in assoc_homs[geneset]:
            res = get_genes(homology)
            for i in res.fetchall():
                _genes.add(i[0])
                _refs.add(i[1])
            _homs.add(homology)
        enrich[geneset] = [_homs, _genes, _refs]

    print("loop2")

    for i in enrich:
        print(i, len(enrich[i][1]), enrich[i])


    # Just for future use
    write_file('enrich/' + filename + '.csv', enrich)
# geneset_odes("filename")
# geneset_odes("gae-onto-onto")

def jac_rank(filename):
    dic = [271955, 218445, 34466, 127372, 614, 219270, 1051, 246704, 34620, 75542, 221932, 34582, 644, 127417, 617, 246616, 218722, 35882, 137565, 1214, 216888, 46984, 553, 839, 14904, 35877, 135191, 34438, 75646, 215984, 921, 75761, 137384, 135194, 271641, 866, 75683, 36285, 218726, 34682, 1766, 216881, 216895, 34302, 845, 34549, 611, 34518, 225899, 271950, 1072, 75604, 34560, 219217, 225900, 218721, 36299, 36287, 34513, 75623, 34904, 34608, 865, 566, 14892, 34625, 137861, 222630, 271725, 34552, 216885, 810, 37188, 552, 271940, 35883, 573, 75657, 355584, 34970, 271727, 34439, 34883, 34786, 271884, 219275, 608, 75644, 271096, 34520, 218958, 1267, 75654, 34314, 246395, 273371, 36320, 267511, 26324, 802]
    vals = read_file("ms-project/grp_ontology.csv")
    ranked_genesets = {}
    # get ranked genesets
    for i in dic:
        x = vals[i]
        y = [i[0] for i in sorted(x.items(), key=lambda xx:getitem(xx[1], 'weight'), reverse=True)[:5]]
        if i in y:
            y.remove(i)
        ranked_genesets[i] = y
    ranked_genesets = dict(sorted(ranked_genesets.items()))
    # write_file('ranking_results/--JCD--Homology.csv', ranked_genesets)
    # sys.exit()

    # get related homology of ranked
    assoc_homs = {}
    for rank in ranked_genesets:
        for geneset in ranked_genesets[rank]:
            if rank in assoc_homs:
                assoc_homs[rank] |= to_onr_set(get_homology(geneset))
            else:
                assoc_homs[rank] = to_onr_set(get_homology(geneset))
    print("loop2")

    enrich = {}
    for geneset in assoc_homs.keys():
        # Iterate over homology
        _homs = set()
        _genes = set()
        _refs = set()
        for homology in assoc_homs[geneset]:
            res = get_genes(homology)
            for i in res.fetchall():
                _genes.add(i[0])
                _refs.add(i[1])
            _homs.add(homology)
        enrich[geneset] = [_homs, _genes, _refs]
    print("loop3")

    for i in enrich:
        print(i, len(enrich[i][1]), enrich[i])

    # Just for future use
    write_file('enrich/' + filename + '.csv', enrich)
# jac_rank("jcd-hom-hom")

def jac_sim(dic, u,v):
    """
        Method calculates jaccard similarity between two nodes
        :param p1: dictionary
        :param p2: node 1
        :param p2: node 2
        """
    try:
        return len(frozenset.intersection(dic[u], dic[v]))/len(frozenset.union(dic[u], dic[v]))
    except TypeError:
        return 0
    except ZeroDivisionError:
        return 0


# Save all gene-set
def save_geneset(filename, attribute='genes'):
    dic = {}
    node_list = []
    result = get_gs_ids()
    for i in result:
        node_list.append(i[0])
    print("Number of nodelist " + str(len(node_list)))
    for i in node_list:
        if attribute == 'genes':
            res = get_homology(i)
        elif attribute == 'ontology':
            res = get_ontology(i)
        pi = [i[0] for i in res.fetchall()]
        dic[i] = frozenset(pi)

    # Just for future use
    # write nodelist to file
    filename = filename+'-'+attribute
    write_file('ms-project/'+filename+'.csv', dic)

""" 
Three types of data extracted so far : 3 Steps
    Step 1 -- Adjacency matrix without weight -- N*N dimensional matrix
    Step 2 -- Adjacency matrix with weight (weight calculated with the jaccard similarity) -- N*N dimensional matrix
    Step 3-- Node features : N*D dimensional matrix -- D number of features(genes), N number of geneset 1/0, 
    -- 1 if geneset contains gene; 0 if geneset does not contain gene
"""

# Step 1: Create and save graph
# Step 2: Create and save adjacency matrix -- weighted and unweighted
def create_graph_n_adj(filename1, filename2, filename3, type='genes'):
    """
    Method creates  adjacency matrix
    1. get required geneset Ids
    2. Save keys into list
    3. for each geneset, get genes.
    4. Create dictionary of geneset and genes: {genesetId: list(genes)}
    :param filename1: filename for graph
    :param filename2: filename for unweighted adjacency matrix
    :param filename3: filename for weighted adjacency matrix
    :return : None
    """

    # read nodelist from file
    if type == 'genes':
        x = read_file('ms-project/genesets-homology.csv')
    elif type == 'ontology':
        x = read_file('ms-project/genesets-ontology.csv')

    node_list = list(x.keys())

    # for graph construction
    edge_list_weigted = []

    # for adjacency matrix
    un_df = pd.DataFrame(0, index=np.arange(len(node_list)), columns=node_list)
    un_df.insert(0, 'piv', node_list)
    un_df.set_index('piv', inplace=True)

    w_df = pd.DataFrame(0, index=np.arange(len(node_list)), columns=node_list)
    w_df.insert(0, 'piv', node_list)
    w_df.set_index('piv', inplace=True)

    for i,j in itertools.product(node_list, node_list):
        if i != j:
            sim = jac_sim(x, i, j)
            edge_list_weigted.append((i, j, {'weight':sim}))
            if sim > 0:
                w_df.loc[i, j] = sim
                un_df.loc[i, j] = 1

    # create networkx graph from nodes and edges created
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list_weigted)

    c1 = nx.to_numpy_array(G)
    print(np.count_nonzero(c1 != 0))
    print(np.count_nonzero(c1 == 0))
    print(c1.shape)

    print(un_df.head(10))
    print(w_df.head(10))

    # write graph to file
    write_file('ms-project/' + filename1 + '.csv', G.adj)
    un_df.to_pickle('ms-project/' + filename2 + '.txt')
    w_df.to_pickle('ms-project/' + filename3 + '.txt')


# Step 3: Create and save feature matrix --N*D
def create_feat_matrix(filename, type='genes'):
    """
    Method creates  feature matrix
    1. get required geneset Ids
    2. Save keys into list
    3. for each geneset, get genes.
    4. Create dictionary of geneset and genes: {genesetId: list(genes)}
    :return : Sparsity of feature matrix
    """

    # if type == 'genes':
    #     x = read_file('ms-project/genesets-homology.csv')
    # elif type == 'ontology':
    #     x = read_file('ms-project/genesets-ontology.csv')
    #
    # node_list = x.keys()
    # feature_list = set()
    # for i in node_list:
    #     feature_list = feature_list.union(set(x[i]))
    #
    # df = pd.DataFrame(0, index=np.arange(len(node_list)), columns=feature_list )
    # df.insert(0, 'piv', node_list)
    # df.set_index('piv', inplace=True)
    # for i in node_list:
    #     for j in x[i]:
    #         df.loc[i, j] = 1
    #
    # print(df.head(10))
    # df.to_pickle('ms-project/' + filename + '.txt')
    df = pd.read_pickle('ms-project/' + filename + '.txt')
    np_array = df.to_numpy()

    # print(scipy.sparse.csr_matrix(np_array))
    # print(np_array.shape)
    # import sys
    # sys.exit()

    # Calcualte sparsity of matrix
    sparsity = (np.prod(np_array.shape) - np.count_nonzero(np_array))/ np.prod(np_array.shape)
    print(sparsity)


# Save all gene-sets
# save_geneset(filename='genesets', attribute='genes')
# save_geneset(filename='genesets', attribute='ontology')

# pair_hom_genes()

create_feat_matrix('feat_genes', type='genes')
create_feat_matrix('feat_ontology', type='ontology')

# create_graph_n_adj("grp_ontology", "adj_ontology_unweighted", "adj_ontology_weighted", type='ontology')

# pair_hom_genes("gcn-hom-hom")