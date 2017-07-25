#!/public/spark-0.9.1/bin/pyspark

import os
import sys
import csv

import datetime

import happybase
import numpy as np
from pynauty import *
import re
import networkx as nx
import ast
from hdfs import Config
from networkx.algorithms.approximation import maximum_independent_set, max_clique
from upc.bsc.Constants import SAMPLES, CHR_MAP, BANDWIDTH_CANDIDATES, THRESHOLD_COUNTERS, SAMPLE_CANCER

sys.path.append("/usr/lib/spark/python")
sys.path.append("/usr/lib/spark/python/lib/py4j-0.10.4-src.zip")
# sys.path.append("/home/jie/d2/spark-0.9.1/python")

# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf

except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)

hdfs_root = 'hdfs://localhost:54310/'
client = Config().get_client('dev')

SparkContext.setSystemProperty('spark.executor.memory', '8g')
conf = SparkConf().setAppName('FrequencyJob').setMaster('local[*]')
sc = SparkContext(conf=conf)


def print_element(el, str_to_print):
    print str(type(el)) + ' ' + str_to_print


def split_into_patterns(line):
    arr = []
    struct = ast.literal_eval(line)
    i = 0
    while i < len(struct) - 1:
        # print 'pattern' + str(pattern)
        pattern = (struct[i], struct[i+1])
        arr.append(str(pattern))
        i += 1
    return arr


def generate_edges(arr):
    if type(arr) == int:
        return []
    edges = []
    for i in range(0, len(arr) - 1):
        for j in range((i+1), len(arr)):
            edges.append((arr[i], arr[j]))
    return edges


def fix_frequency(connected_embeddings):
    flat_list = [item for sublist in connected_embeddings for item in sublist]
    # print flat_list
    flat_set = set(flat_list)
    # print flat_set
    nx_g = nx.Graph()
    nx_g.add_nodes_from(flat_set)
    lines_inter = sc.parallelize(connected_embeddings)
    #print lines_inter.collect()
    edges = lines_inter.flatMap(lambda arr: generate_edges(arr))
    edges = edges.distinct()
    #edges.foreach(lambda edge: nx_g.add_edge(edge[0], edge[1]))
    nx_g.add_edges_from(edges.collect())
    # print 'nodes ' + str(len(list(nx_g.nodes())))
    # print 'edges ' + str(len(list(nx_g.edges())))
    connected = list(nx.connected_component_subgraphs(nx_g))
    freq = 0
    for graph in connected:
        freq += len(maximum_independent_set(graph))
    return freq

connection = happybase.Connection(host='localhost')
sample_pattern_table = connection.table('sample_pattern')

for bandwidth in BANDWIDTH_CANDIDATES:
    for sample in SAMPLES:
        for threshold_counter in THRESHOLD_COUNTERS:
            for size in range(2, 5):
                print 'BANDWIDTH ' + str(bandwidth) + ', THRESHOLD ' + str(threshold_counter) + ', SAMPLE ' + sample + ', SIZE ' + str(size)
                # try to read from hdfs
                folder = 'subgraphs/b' + str(int(bandwidth)) + '/' + sample + '/t' + str(threshold_counter) + '/' + str(size)
                fnames = client.list(folder)
                for fname in fnames:
                    with client.read(folder + '/' + fname, encoding='utf-8') as reader:
                        for line in reader:
                            struct = ast.literal_eval(line)
                            freq_inter = fix_frequency(struct[1][0][1].values())
                            print 'pattern:' + struct[0] + ', inter freq: ' + str(struct[1][0][0]) + ', fixed: ' + str(freq_inter)

                            # print 'intra' + str(struct[1][1][1].values())
                            freq_intra = fix_frequency(struct[1][1][1].values())
                            print 'pattern:' + struct[0] + ', intra freq: ' + str(struct[1][1][0]) + ', fixed: ' + str(freq_intra)
                            row_key = 'b' + str(int(bandwidth)) + 't' + str(threshold_counter) + 's' + sample + 'p' + struct[0].strip()
                            values = { b'f:fix_freq_inter': str(freq_inter).encode('utf-8'), b'f:fix_freq_intra': str(freq_intra).encode('utf-8')}
                            sample_pattern_table.put(row_key.encode('utf-8'), values)

                # lines = sc.textFile(hdfs_root + folder)
                # separate_patterns = lines.flatMap(lambda line: split_into_patterns(line))
                # separate_patterns = separate_patterns.map(lambda pattern: ast.literal_eval(pattern))
                #
                # separate_patterns.foreach(lambda el: print_element(el, 'element ' + str(el)))

