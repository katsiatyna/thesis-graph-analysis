#!/public/spark-0.9.1/bin/pyspark

import os
import sys
import re
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

sys.path.append("/usr/lib/spark/python")
sys.path.append("/usr/lib/spark/python/lib/py4j-0.10.4-src.zip")
# sys.path.append("/home/jie/d2/spark-0.9.1/python")

# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark import SparkContext
    from pyspark.streaming import StreamingContext
    from pyspark.streaming.kafka import KafkaUtils

except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)


def map_to_graph(combination):
    # edges are list of tuples
    from SubgraphCollection import SubgraphCollection
    from VsigramGraph import VsigramGraph
    p = re.compile('\(\d+, \d+\)')
    # only taking value
    original_edges_str = combination[1]
    # print 'COMBINATIONS ARE ' + str(original_edges_str) + ' ' + str(type(original_edges_str))
    matched = p.findall(original_edges_str)
    #print 'MATCHES ARE ' + str(matched)
    p = re.compile('\d+')
    original_edges = list()
    for match in matched:
        vertices = p.findall(match)
        original_edges.append((int(vertices[0]), int(vertices[1])))
    original_vertices = set()
    for edge in original_edges:
        # print 'EDGE is ' + str(edge) + ' ' + str(type(edge))
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
    nx_g = nx.Graph()
    nx_g.add_edges_from(new_edges)
    if nx.is_connected(nx_g):
        subgraph_collection = SubgraphCollection(v_g.label_arr, subgraphs=[v_g], freq=1)
    else:
        subgraph_collection = None
    return (subgraph_collection.label if subgraph_collection is not None else '', subgraph_collection)


def update_subgraph_freq(a, b):
    from SubgraphCollection import SubgraphCollection
    freq_object = SubgraphCollection(label=a.label)
    freq_object.subgraphs = a.subgraphs + b.subgraphs
    freq_object.freq = a.freq + b.freq
    return freq_object


def filter_by_connected(edges):
    # create networkx graph
    nx_g = nx.Graph()
    nx_g.add_edges_from(edges)
    return nx.is_connected(nx_g)

def main(ssc):
    directKafkaStream = KafkaUtils.createDirectStream(ssc, ['subgraphs'], {"metadata.broker.list": 'localhost:9092'})
    # create a graph from each list
    rdd_of_graphs = directKafkaStream.map(lambda combination: map_to_graph(combination))
    rdd_filtered = rdd_of_graphs.filter(lambda x: x[1] is not None)
    counts_by_label = rdd_filtered.reduceByKey(lambda a, b: update_subgraph_freq(a, b))
    counts_by_label.pprint()
    # # counts_by_label_list = counts_by_label.collect()
    # # for element in counts_by_label_list:
    #     # print element[0] + ': FREQ is ' + str(element[1].freq)
    # counts_by_label.pprint()
    ssc.start()
    ssc.awaitTermination()

if __name__ == "__main__":
    sc = SparkContext("local[*]", "SubgraphMining",
                      pyFiles=['/home/kkrasnas/PycharmProjects/thesis-graph-analysis/upc/bsc/grahanalysis/SubgraphCollection.py',
                               '/home/kkrasnas/PycharmProjects/thesis-graph-analysis/upc/bsc/grahanalysis/VsigramGraph.py'])
    ssc = StreamingContext(sc, 5)
    main(ssc)

