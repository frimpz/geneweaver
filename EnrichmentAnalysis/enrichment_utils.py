import csv
import ctypes as ct
import re

csv.field_size_limit(int(ct.c_ulong(-1).value//2))

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


def read_file_2(filename):
    with open(filename, 'r', encoding="utf8") as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        my_dict = dict(reader)
        my_dict = {u: eval(v) for u, v in my_dict.items()}
        return my_dict


def write_file(filename, dic):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=':')
        for key, value in dic.items():
            writer.writerow([key, value])



# get codename in parenthesis
def get_code_name(s):
    res = [i.strip("()") for i in re.findall(r'\(.*?\)', s)]
    return res


def reduce_genesets():
    files = ["gae-hom-hom", "gae-hom-onto", "gae-onto-onto", "gae-onto-hom", "jcd-hom", "jcd-onto", "selected_genesets"]

    possible_empty = set()
    for i in files:
        main = read_file("../enrich2/"+i+".csv")

        for j in main:
            if len(main[j]) == 0:
                possible_empty.add(j)

    main = read_file_2("../neighbours/neig_len_hom.csv")
    for i in main:
        if int(main[i]) == 0:
            possible_empty.add(i)

    main = read_file_2("../neighbours/neig_len_onto.csv")
    for i in main:
        if int(main[i]) == 0:
            possible_empty.add(i)


    for i in files:
        temp = {}
        main = read_file("../enrich2/" + i + ".csv")
        for j in main:
            if j not in possible_empty:
                temp[j] = main[j]
        write_file("../enrich_red/" + i + ".csv", temp)


    for i in files:
        main = read_file_2("../enrich_red/" + i + ".csv")
        print(len(main))

def to_uppercases():
    files = ["gae-hom-hom", "gae-hom-onto", "gae-onto-onto", "gae-onto-hom", "jcd-hom", "jcd-onto", "selected_genesets"]

    for i in files:
        main = read_file_2("../enrich_red/"+i+".csv")


        for j in main:
            main[j] = [x.upper() for x in main[j]]
            print(j, main[j])

        write_file("../enrich_red/" + i + ".csv", main)


def just_read():
    files = ["gae-hom-hom", "gae-hom-onto", "gae-onto-onto", "gae-onto-hom", "jcd-hom", "jcd-onto",
             "selected_genesets"]

    for i in files:
        main = read_file_2("../enrich_red/" + i + ".csv")

        for j in main:
            print(j, main[j])


def merge_all_perms():
    keys = ["gae-hom-hom-gae-hom-onto", "gae-hom-hom-gae-onto-onto",
    "gae-hom-hom-gae-onto-hom", "gae-hom-hom-jcd-hom",
    "gae-hom-hom-jcd-onto",

    "gae-hom-onto-gae-onto-onto", "gae-hom-onto-gae-onto-hom",
    "gae-hom-onto-jcd-hom", "gae-hom-onto-jcd-onto",

    "gae-onto-onto-gae-onto-hom", "gae-onto-onto-jcd-hom",
    "gae-onto-onto-jcd-onto",

    "gae-onto-hom-jcd-hom", "gae-onto-hom-jcd-onto",

    "jcd-hom-jcd-onto"]

    files = ['GO_Biological_Process_2018.csv', 'KEGG_2019_Human.csv']

    merged = {}
    for i in files:
        file = read_file_2("../perms2/"+i)
        for j in keys:
            if j in merged:
                merged[j].extend(file[j])
            else:
                merged[j] = file[j]


    for i in merged:
        print(len(merged[i]))

    write_file("../perms2/merged.csv", merged)

# merge_all_perms()
# reduce_genesets()

# to_uppercases()

