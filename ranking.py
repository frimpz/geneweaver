# Ranking Similarity

from utils import get_geneset, point_ranking, get_all_geneset, get_index, write_file
import pandas as pd


def create_ranking(filename, data, _ids=None, top_k=5, file_type='xlsx'):

    if _ids is None:
        gs = get_all_geneset()
    else:
        gs = _ids

    ind = list(range(0, data.shape[0]))
    gensets = get_geneset(ind)

    rankings = []
    rankings_to_df = []

    for i in gs:
        index = get_index(i)
        point_embeddings = [data[index]]
        pr = point_ranking(point_embeddings, data, ind, gensets, top_k)

        rankings.append(pr)
        for k in pr:
            zone = [i, k[1], k[2]]
            rankings_to_df.append(zone)
        cols = ['Node', 'Similar', 'rank-distance']

    rank_df = pd.DataFrame(rankings_to_df, columns=cols)
    rank_df = rank_df.set_index(['Node', 'Similar'], inplace=False)

    if file_type == 'xlsx':
        writer = pd.ExcelWriter(filename)
        rank_df.to_excel(writer)
        writer.save()
    elif file_type == 'csv':
        rank_df.to_csv(filename, sep='\t')


def create_ranking_(filename, data, _ids=None, top_k=5):

    dic = {}
    ind = list(range(0, data.shape[0]))
    gensets = get_geneset(ind)
    _ids = get_all_geneset()

    for geneset in _ids:
        index = get_index(geneset)
        point_embeddings = [data[index]]
        pr = point_ranking(point_embeddings, data, ind, gensets, top_k)
        dic[geneset] = [i[1] for i in pr if i[1]!=geneset][:4]

    write_file(filename, dic)


from random import shuffle
x = list(get_all_geneset())
print(x)
shuffle(x)
print(x[:100])

