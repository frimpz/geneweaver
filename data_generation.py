# This file is used to create the data for the node clustering.
# G(V,E) -- undirected graph
# from itertools import chain
import itertools
# todo: continue documentation

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


def get_gs_ids():
    """
    Method returns the genesets in the graph
    You can ask Dr. Baker for curation group, he stated that it will reduce the total number of nodes
    : return : cursor of geneset IDS
    """
    with PooledCursor() as cursor:
        cursor.execute(
            '''
            SELECT gs.gs_id 
            FROM production.geneset gs 
            WHERE gs.cur_id=3 
            AND gs.gs_status 
            NOT LIKE 'de%';
            '''
        )
    return cursor


def get_genes(geneset):
    """
    Method returns the genes for each geneset selected
    :param p1 : geneset Id
    : return : cursor of geneset IDS
    """
    with PooledCursor() as cursor:
        cursor.execute(
            '''
            SELECT distinct gs.ode_gene_id
            FROM extsrc.geneset_value AS gs
            WHERE gs.gs_id = %s;
            ''',
            (geneset,)
        )
    return cursor


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


""" 
Three types of data extracted so far : 3 Steps
    Step 1 -- Adjacency matrix without weight -- N*N dimensional matrix
    Step 2-- Node features : N*D dimensional matrix -- D number of features(genes), N number of geneset 1/0, 
    -- 1 if geneset contains gene; 0 if geneset does not contain gene
"""

# Step 1: Create and save adjacency matrix -- weighted
def create_adj():
    """
    Method creates  adjacency matrix
    1. get required geneset Ids
    2. Save keys into list
    3. for each geneset, get genes.
    4. Create dictionary of geneset and genes: {genesetId: list(genes)}
    :return : None
    """

    dic = {}
    node_list = []
    result = get_gs_ids()
    for i in result:
        node_list.append(i[0])
        print(i)
    print("Number of nodelist "+ str(len(node_list)))
    for i in node_list:
        res = get_genes(i)
        pi = [i[0] for i in res.fetchall()]
        dic[i] = frozenset(pi)

    # Just for future use
    # write nodelist to file
    write_file('node_graph_2.csv', dic)

    # read nodelist from file
    # x = read_file('node_graph.csv')

    # get edges -- calculate weights also
    edge_list = []
    for i,j in itertools.product(node_list, node_list):
        if i != j:
            edge_list.append((i,j, {'weight':jac_sim(dic, i, j)}))

    # create networkx graph from nodes and edges created
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    c = nx.to_numpy_array(G)
    print(np.count_nonzero(c != 0))
    print(np.count_nonzero(c == 0))
    print(c.shape)

    # write graph to file
    write_file('adj_2.csv', G.adj)


# Step 2: Create and save feature matrix --N*D
def create_feat_matrix():
    """
    Method creates  feature matrix
    1. get required geneset Ids
    2. Save keys into list
    3. for each geneset, get genes.
    4. Create dictionary of geneset and genes: {genesetId: list(genes)}
    :return : Sparsity of feature matrix
    """
    x = read_file('node_graph_2.csv')

    node_list = x.keys()
    feature_list = set()
    for i in node_list:
        for j in x[i]:
            feature_list.add(j)
    df = pd.DataFrame(0, index=np.arange(len(node_list)), columns=feature_list )
    df.insert(0, 'piv', node_list)
    df.set_index('piv', inplace=True)
    for i in node_list:
        for j in x[i]:
            df.loc[i, j] = 1

    df.to_pickle("node_graph_np_2.txt")
    # df = pd.read_pickle("node_graph_np.txt")
    np_array = df.to_numpy()

    # print(scipy.sparse.csr_matrix(np_array))
    # print(np_array.shape)
    # import sys
    # sys.exit()

    # Calcualte sparsity of matrix
    sparsity = (np.prod(np_array.shape) - np.count_nonzero(np_array))/ np.prod(np_array.shape)


create_adj()
create_feat_matrix()