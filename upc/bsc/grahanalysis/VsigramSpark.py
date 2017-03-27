#!/public/spark-0.9.1/bin/pyspark

import os
import sys
import csv
import itertools
import numpy as np
from pynauty import *
import re
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import pydot
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot


# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
#os.environ['SPARK_HOME'] = "/usr/lib/spark"
# os.environ['SPARK_HOME'] = "/home/jie/d2/spark-0.9.1"
# Append to PYTHONPATH so that pyspark could be found
from upc.bsc.grahanalysis.model.SubgraphCollection import SubgraphCollection
from upc.bsc.grahanalysis.model.VsigramGraph import VsigramGraph

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


def map_csv_to_edges_list(path='/home/kkrasnas/Documents/thesis/pattern_mining/new_assignment.csv'):
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['pos_1', 'pos_2'])
        next(csvfile)
        positions = list()
        edges_set = set()
        edges = list()
        for row in reader:
            positions.append((row['pos_1'], row['pos_2']))

        vertices_set = set()
        for position in positions:
            vertices_set.add(position[0])
            vertices_set.add(position[1])
        vertices_list = list(vertices_set)
        g = Graph(len(vertices_list))

        # ORIGINAL EDGES ARE ALL SORTED MIN -> MAX
        for edge in positions:
            # always minIndex -> maxIndex to avoid duplicate edges
            if vertices_list.index(edge[0]) < vertices_list.index(edge[1]):
                g.connect_vertex(vertices_list.index(edge[0]), vertices_list.index(edge[1]))
                edges_set.add((vertices_list.index(edge[0]), vertices_list.index(edge[1])))
            else:
                g.connect_vertex(vertices_list.index(edge[1]), vertices_list.index(edge[0]))
                edges_set.add((vertices_list.index(edge[1]), vertices_list.index(edge[0])))
        edges = list(edges_set)
        return edges


def map_to_graph(combination):
    # edges are list of tuples

    # this is going to be the key
    canonical = ''
    original_edges = combination
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
    v_g = VsigramGraph(g, 'no_hash_needed', certificate(g), canon_label(g),
                       edges=new_edges, vertices=range(len(original_vertices)),
                       orig_edges=original_edges, orig_vertices=original_vertices)
    subgraph_collection = SubgraphCollection(v_g.label_arr, subgraphs=[v_g], freq=1)
    return (subgraph_collection.label, subgraph_collection)


def update_subgraph_freq(a, b):
    freq_object = SubgraphCollection(label=a.label)
    freq_object.subgraphs = a.subgraphs + b.subgraphs
    freq_object.freq = a.freq + b.freq
    return freq_object


conf = SparkConf().setAppName('SubgraphMining').setMaster('local')
sc = SparkContext(conf=conf)

# load the edges and deduplicate them
edges = map_csv_to_edges_list()


for i in range(2, len(edges) + 1):
    # combinations = itertools.combinations(range(len(edges)), i)
    combinations = itertools.combinations(edges, i)
    # print list(combinations)
    combinations_list = list(combinations)
    # print combinations_list
    rdd_size = sc.parallelize(combinations_list)
    rdd_list = rdd_size.collect()
    # print 'rdd: ' + str(rdd_list)

    # create a graph from each list
    rdd_of_graphs = rdd_size.map(lambda combination: map_to_graph(combination))
    counts_by_label = rdd_of_graphs.reduceByKey(lambda a, b: update_subgraph_freq(a, b))
    counts_by_label_list = counts_by_label.collect()
    for element in counts_by_label_list:
        print element[0] + ': FREQ is ' + str(element[1].freq)
