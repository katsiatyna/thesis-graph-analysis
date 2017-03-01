#!/public/spark-0.9.1/bin/pyspark

import os
import sys
import numpy as np
from pynauty import *
import re
import networkx as nx
from model.ARule import *
from model.FIGraph import *
from model.ARGraph import *
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

def create_ar_graphs(row):
    pass

def deduplicate_freq_itemsets(rule):
    #extract edges, reorder vertices inside, then reorder edges
    edges = rule.ante.split(',') if rule.ante is not None else []
    edges.append(rule.conseq)
    print edges
    reordered_edges = list()
    for edge in edges:
        if edge != '':
            vertices = edge.split('.')
            vertices = sorted(map(int,vertices))
            reordered_edges.append(''+str(vertices[0])+'.'+str(vertices[1]))
    reordered_edges = sorted(reordered_edges)
    #hash e.g. 31.23.34.5
    hash = str(len(reordered_edges)) + ':' + ':'.join(reordered_edges)
    print hash, rule.support_abs, rule.support_rel
    return (hash,rule)


def map_arule_to_ar_graphs(rule):
    # RHS + full
    rhs_vertices = rule.conseq.split('.') if rule.conseq is not None else []
    full_vertices = list(rhs_vertices)
    rhs_set = set(map(int, rhs_vertices))
    full_set = set(map(int, full_vertices))
    rhs_edges = list()
    rhs_edges.append(sorted(map(int, rhs_vertices)))
    full_edges = list()
    full_edges.append(sorted(map(int, full_vertices)))

    lhs_edges_str = rule.ante.split(',') if rule.ante is not None else []
    lhs_set = set()
    lhs_edges = list()
    for edge in lhs_edges_str:
        vertices = edge.split('.')
        if len(vertices) > 1:
            lhs_set |= set(map(int, vertices))
            full_set |= set(map(int, vertices))
            lhs_edges.append(sorted(map(int, vertices)))
            full_edges.append(sorted(map(int, vertices)))
    full_set = sorted(list(full_set))
    lhs_set = sorted(list(lhs_set))
    rhs_set = sorted(list(rhs_set))
    g_full = Graph(len(full_set))
    for edge in full_edges:
        if len(edge) > 0:
            g_full.connect_vertex(full_set.index(edge[0]), full_set.index(edge[1]))
    g_rhs = Graph(len(rhs_set))
    for edge in rhs_edges:
        if len(edge) > 0:
            g_rhs.connect_vertex(rhs_set.index(edge[0]), rhs_set.index(edge[1]))
    g_lhs = Graph(len(lhs_set))
    for edge in lhs_edges:
        if len(edge) > 0:
            g_lhs.connect_vertex(lhs_set.index(edge[0]), lhs_set.index(edge[1]))

    cert = certificate(g_lhs) + certificate(g_rhs) + certificate(g_full)
    return (cert, ARGraph(graph_lhs=g_lhs, graph_rhs=g_rhs, graph_full=g_full, count=1, full_tids=rule.full_tids, lhs_tids=rule.lhs_tids))

def reduce_by_shapes(a, b):

    return ARGraph(a.graph_lhs, a.graph_rhs, a.graph_full, a.count + b.count, full_tids=a.full_tids.union(b.full_tids),
                                                   lhs_tids=a.lhs_tids.union(b.lhs_tids))


def add_support_and_conf(val, n):
    val[1].support_abs = len(val[1].full_tids)
    val[1].support_rel = len(val[1].full_tids) / float(n)
    val[1].conf = float(len(val[1].full_tids)) / float(len(val[1].lhs_tids))
    return val


def draw_graphs(graphs):
    i = 0
    for example in graphs:

        #convert to networkx graph
        print (i,example[1].fi_graph)
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


conf = SparkConf().setAppName('ARulesShapes').setMaster('local')
sc = SparkContext(conf=conf)
size = 1980
lines = sc.textFile("/home/kkrasnas/Documents/thesis/pattern_mining/tables/rules_ext_sample.csv")
header = lines.first() #extract header
lines = lines.filter(lambda row: row != header)   #filter out header
print lines.count()

# convert each line to a rule
rules = lines.map(lambda row: ARule(row, size, False))

# hash each rule to define which represent EXACT SAME graphs
#hashed_rules = rules.map(lambda rule: deduplicate_freq_itemsets(rule))

# trim the rules by leaving only distinct itemsets - only for frequent itemsets
#deduplicated_rules = hashed_rules.reduceByKey(lambda a,b: a)
#print deduplicated_rules.count()

# map: key - graph hash, value - graph itself + count + support
hashedGraphs = rules.map(lambda rule: map_arule_to_ar_graphs(rule))

# reduce with adding absolute support
group_by_shapes = hashedGraphs.reduceByKey(lambda a, b: reduce_by_shapes(a,b))

# calculate relative support with absolute and size of the dataset
ar_with_support = group_by_shapes.map(lambda val: add_support_and_conf(val, size))

all_graphs = ar_with_support.collect()
#sum = 0
for gr in all_graphs:
    print gr[0], gr[1].support_abs, gr[1].support_rel, gr[1].conf
#    sum += gr[1].count
#print sum
#draw_graphs(all_graphs)

#fi_with_support.saveAsTextFile('/home/kkrasnas/Documents/thesis/pattern_mining/tables/test_res_sup1.out')




