#!/public/spark-0.9.1/bin/pyspark

import os
import sys
from pynauty import *
import re
from ARule import *
from GraphCount import *
import networkx as nx
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

import numpy as np

def myFunc(row):
    #parse line
    rule = ARule(row)
    #print rule
    full_vertices = rule.conseq.split('.') if rule.conseq != None else []
    full_set = set(map(int, full_vertices))
    full_edges = list()
    full_edges.append(sorted(map(int, full_vertices)))


    edges = rule.ante.split(',') if rule.conseq != None else []
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

conf = SparkConf().setAppName('PyTest').setMaster('local')
sc = SparkContext(conf=conf)
lines = sc.textFile("/home/kkrasnas/Documents/thesis/pattern_mining/tables/rules_clean.csv")

#map: key - graph hash, value - graph itself
hashedGraphs = lines.map(lambda row: myFunc(row) )
grouping = hashedGraphs.reduceByKey(lambda a, b: GraphCount(a.graph, a.count + b.count))
#all_graphs = grouping.collect()
#example = all_graphs[0]
#convert to networkx graph
#G = nx.Graph()

grouping.saveAsTextFile('/home/kkrasnas/Documents/thesis/pattern_mining/tables/test_res_full6.out')


