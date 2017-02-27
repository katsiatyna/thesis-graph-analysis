#!/public/spark-0.9.1/bin/pyspark

import os
import sys
import numpy as np
from pynauty import *
import re
import networkx as nx
from model.ARule import *
from model.GraphCount import *
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

except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)




def map_freq_itemsets(row):
    #parse line
    rule = ARule(row)
    #print rule
    full_vertices = rule.conseq.split('.') if rule.conseq != None else []
    full_set = set(map(int, full_vertices))
    full_edges = list()
    full_edges.append(sorted(map(int, full_vertices)))


    edges = rule.ante.split(',') if rule.ante != None else []
    vertices_set = set()
    for edge in edges:
        vertices = edge.split('.')
        if len(vertices) > 1:
            full_set |= set(map(int, vertices))
            full_edges.append(sorted(map(int, vertices)))
    full_set = sorted(list(full_set))
    g = Graph(len(full_set))
    for edge in full_edges:
        if len(edge) > 0:
            g.connect_vertex(full_set.index(edge[0]), full_set.index(edge[1]))
    cert = certificate(g)
    #print (row['1:nrow(p3)'], full_set, cert)
    return (cert,GraphCount(g, 1))


def draw_graphs(all_graphs):
    i = 0
    for example in all_graphs:

        #convert to networkx graph
        print (i,example[1].graph)
        nx_g = nx.MultiGraph()
        nx_g.add_nodes_from(example[1].graph._get_adjacency_dict().iterkeys())
        for key in example[1].graph._get_adjacency_dict().iterkeys():
            for val in example[1].graph._get_adjacency_dict()[key]:
                nx_g.add_edge(key,val)
        plt.figure(i)
        #nx.draw_circular(nx_g, with_labels=False, node_color='blue')
        #pos = graphviz_layout(nx_g)
        #nx.draw(nx_g, pos)
        write_dot(nx_g,''+i+'graph.dot')
        i += 1
    #plt.interactive(False)
    #plt.show()


conf = SparkConf().setAppName('PyTest').setMaster('local')
sc = SparkContext(conf=conf)
lines = sc.textFile("/home/kkrasnas/Documents/thesis/pattern_mining/tables/rules_clean.csv")

#map: key - graph hash, value - graph itself
hashedGraphs = lines.map(lambda row: map_freq_itemsets(row))
grouping = hashedGraphs.reduceByKey(lambda a, b: GraphCount(a.graph, a.count + b.count))
all_graphs = grouping.collect()
#for gr in all_graphs:
 #   print gr[0], gr[1].count
#draw_graphs(all_graphs)

#grouping.saveAsTextFile('/home/kkrasnas/Documents/thesis/pattern_mining/tables/test_res_full7.out')




