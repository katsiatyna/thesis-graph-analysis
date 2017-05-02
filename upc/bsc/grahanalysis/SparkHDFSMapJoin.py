#!/public/spark-0.9.1/bin/pyspark

import os
import sys
import csv
import numpy as np
from pynauty import *
import re
import networkx as nx
from hdfs import Config


# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
#os.environ['SPARK_HOME'] = "/usr/lib/spark"
# os.environ['SPARK_HOME'] = "/home/jie/d2/spark-0.9.1"
# Append to PYTHONPATH so that pyspark could be found
# from SubgraphCollection import SubgraphCollection
# from VsigramGraph import VsigramGraph

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


def get_subgraph_hash(edges_indexes):
    # ORDER INSIDE EDGES IS ALREADY DONE
    # NOW ONLY NEED TO ORDER EDGES BETWEEN THEMSELVES
    hash_str = ''
    # rearrange the edges
    edges_indexes = sorted(edges_indexes)  # sorts on the first elements first and then on second
    for edge in edges_indexes:
        hash_str += str(edge) + ';'
    return hash_str


def map_to_tuple_with_hash(combination):
    # edges are list of indexes
    # this is going to be the key
    if type(combination) is int:
        combination = [combination]
    return (get_subgraph_hash(combination), combination)


def transform_to_original_edges(edges_indexes, positions_list):
    result_edges = list()
    for index in edges_indexes:
        result_edge = (positions_list[index[0]], positions_list[index[1]])
        result_edges.append(result_edge)
    return result_edges


def map_to_graph(combination, edges, positions_list):
    # edges are list of tuples

    # this is going to be the key
    original_edges = list()
    if type(combination) is int:
        combination = [combination]
    for ind in combination:
        original_edges.append(edges.value[ind])
    original_vertices = set()
    for edge in original_edges:
        original_vertices.add(edge[0])
        original_vertices.add(edge[1])
    original_vertices = list(original_vertices)
    g = Graph(len(original_vertices))
    new_edges = list()
    for edge in original_edges:
        g.connect_vertex(original_vertices.index(edge[0]), original_vertices.index(edge[1]))
        new_edges.append((original_vertices.index(edge[0]), original_vertices.index(edge[1])))
    # v_g = VsigramGraph(g, None, label_arr=canon_label(g), orig_edges=original_edges)
    # subgraph_collection = SubgraphCollection(v_g.label_arr, subgraphs=[v_g], freq=1)
    return (canon_label(g), (1, [transform_to_original_edges(original_edges, positions_list)]))


def update_subgraph_freq(a, b):
    # freq_object = SubgraphCollection(label=a.label)
    # freq_object.subgraphs = a.subgraphs + b.subgraphs
    # freq_object.freq = a.freq + b.freq
    return (a[0] + b[0], a[1] + b[1])


def filter_by_connected(edges_indexes, orig_edges):
    # create networkx graph
    first_element = True
    edge_new = edges_indexes[len(edges_indexes) - 1]
    edges_old = list(edges_indexes[0:len(edges_indexes) - 1]) if type(edges_indexes[0:len(edges_indexes) - 1]) is list \
        else [edges_indexes[0:len(edges_indexes) - 2]]
    # print str(edges_old) + str(type(edges_old))
    # print str(edge_new) + str(type(edge_new))
    if first_element:
        # print 'INDEXES: ' + str(edges_indexes) + str(type(edges_indexes))
        # print 'OLD: ' + str(edges_old) + str(type(edges_old))
        # print 'NEW: ' + str(edge_new) + str(type(edge_new))
        # print str(edge_new in edges_old)
        first_element = False
    if edge_new in edges_old:
        # print 'REPEATING EDGE'
        return False
    edges_list = list()
    for ind in list(edges_indexes):
        edges_list.append(orig_edges[ind])
    # print edges_list
    nx_g = nx.Graph()
    nx_g.add_edges_from(edges_list)
    return nx.is_connected(nx_g)


def mapToList(edges_indexes):
    tupl_to_list = list(edges_indexes[0]) if type(edges_indexes[0]) is list else [edges_indexes[0]]
    tupl_to_list.append(edges_indexes[1])
    # print 'LIST: ' + str(tupl_to_list)
    return tupl_to_list


def join_connected_edges(combination, original_edges):
    edges_indexes_list = list(combination) if type(combination) is list else [combination]
    result_list = []
    combination_edges = []
    vertices_set = set()
    for edge_index in edges_indexes_list:
        edge_current = original_edges.value[edge_index]
        combination_edges.append(edge_current)
        vertices_set.add(edge_current[0])
        vertices_set.add(edge_current[1])
    for edge in original_edges.value:
        if edge not in combination_edges and (edge[0] in vertices_set or edge[1] in vertices_set):
            new_list = edges_indexes_list + [original_edges.value.index(edge)]
            # print new_list
            result_list.append(new_list)
    # print 'LISTS: ' + str(result_list)
    return result_list

sample = 'ea1cac20-88c1-4257-9cdb-d2890eb2e123'
hdfs_root = 'hdfs://localhost:54310/'
client = Config().get_client('dev')
print 'Deleting HDFS directory...'
client.delete('subgraphs/' + sample, recursive=True)
conf = SparkConf().setAppName('SubgraphMining').setMaster('local[*]')
sc = SparkContext(conf=conf)

# load the edges and deduplicate them
# edges = map_csv_to_edges_list(path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv')



# try to read from hdfs
lines = sc.textFile(hdfs_root + 'samples/' + sample + '/' + sample + '_new_assignment.csv')
header = lines.first()  # extract header
lines = lines.filter(lambda row: row != header)   # filter out header
positions_rdd = lines.map(lambda line: line.split(','))
print positions_rdd.collect()
positions_combined = lines.flatMap(lambda line: line.split(','))
positions_distinct = positions_combined.distinct()
positions_distinct_list = positions_distinct.collect()
edges_rdd = positions_rdd.map(lambda positions: [positions_distinct_list.index(positions[0])
                                                 if positions_distinct_list.index(positions[0]) < positions_distinct_list.index(positions[1])
                                                 else positions_distinct_list.index(positions[1]),
                                                 positions_distinct_list.index(positions[1])
                                                 if positions_distinct_list.index(positions[0]) < positions_distinct_list.index(positions[1])
                                                 else positions_distinct_list.index(positions[0])])
edges_rdd = edges_rdd.map(lambda edge: (str(edge[0]) + ':' + str(edge[1]), (edge, 1)))
edges_rdd = edges_rdd.reduceByKey(lambda edge_kv1, edge_kv2: (edge_kv1[0], edge_kv1[1] + edge_kv2[1]) )
edges_rdd_list = edges_rdd.collect()
edges = [tuple(item[1][0]) for item in edges_rdd_list]
print len(edges)
edges_list = sc.broadcast(edges)

rdd_1 = sc.parallelize(range(len(edges)))
rdd_of_graphs_1 = rdd_1.map(lambda combination: map_to_graph(combination, edges_list, positions_distinct_list))
#     rdd_filtered = rdd_of_graphs.filter(lambda x: x[1] is not None)
counts_by_label_1 = rdd_of_graphs_1.reduceByKey(lambda a, b: update_subgraph_freq(a, b))
counts_by_label_list_1 = counts_by_label_1.collect()
counts_by_label_1.saveAsTextFile(hdfs_root + 'subgraphs/' + sample + '/' + str(1))
rdd_last = rdd_1

for i in range(2, 4):
    print 'SIZE ' + str(i)
    # for each element in rdd_1 create a list and add to new rdd
    rdd_next = rdd_last.flatMap(lambda combination: join_connected_edges(combination, edges_list))
    print rdd_next.first()
    rdd_next = rdd_next.map(lambda combination: map_to_tuple_with_hash(combination))
    print rdd_next.count()
    print rdd_next.first()
    rdd_next = rdd_next.reduceByKey(lambda a, b: a)
    print rdd_next.count()
    rdd_next = rdd_next.map(lambda x: x[1])
    print rdd_next.first()
    rdd_last = rdd_next
    # create a graph from each list
    rdd_of_graphs = rdd_next.map(lambda combination: map_to_graph(combination, edges_list, positions_distinct_list))
    print rdd_of_graphs.first()
    counts_by_label = rdd_of_graphs.reduceByKey(lambda a, b: update_subgraph_freq(a, b))
    #print counts_by_label.first()
    counts_by_label.saveAsTextFile(hdfs_root + 'subgraphs/' + sample + '/' + str(i))
