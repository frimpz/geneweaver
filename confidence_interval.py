# Confidence Interval Claculation

import csv
import sys
import ctypes as ct
import random

import math
from matplotlib.patches import Rectangle

csv.field_size_limit(int(ct.c_ulong(-1).value//2))
import gseapy as gp
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as spy
import statistics
import scipy.stats as st


def read_file_2(filename):
    with open(filename, 'r', encoding="utf8") as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        my_dict = dict(reader)
        my_dict = {u: eval(v) for u, v in my_dict.items()}
        return my_dict



def find_values(file_name, alpha=0.05):

    x = read_file_2(file_name)

    excel_rows = [['Method', 'sample-size', 'PT. Est', 'lower CI', 'upper CI']]
    for i in x:
        method = i
        size = len(x[i])
        t_dist = st.t.ppf(1-(alpha/2), size-1)
        pt_est = statistics.mean(x[i])
        std = statistics.stdev(x[i])
        lower_ci = max(pt_est-t_dist*(std/math(size)), 0)
        upper_ci = pt_est + t_dist*(std/math(size))

        excel_rows.append(method, size, pt_est, lower_ci, upper_ci)

    df = pd.DataFrame.from_records(excel_rows[1:], columns=excel_rows[0])
    return df


print(find_values(file_name, alpha=0.05))