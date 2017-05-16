import datetime
from hdfs import Config
import ast
import csv
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set, max_clique
from networkx.algorithms import maximal_independent_set
import igraph as ig
import matplotlib.pyplot as plt
import sys
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


def write_init_files(client, sample):
    # write the assignment file
    with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv', 'rw') as csvfile:
        client.delete('sample', recursive=True)
        client.delete('samples/new_assignment_separate.csv', recursive=True)
        client.write('samples/new_assignment_separate.csv', csvfile)
    file_name = sample + '_new_assignment.csv'
    metrics_file_name = sample + '_metrics.csv'
    with open('/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/' + file_name,
              'rw') as csvfile:
        client.delete('samples/' + sample + '/' + file_name, recursive=True)
        client.write('samples/' + sample + '/' + file_name, csvfile)
    with open('/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/' + metrics_file_name,
              'rw') as csvfile:
        client.delete('samples/' + sample + '/' + metrics_file_name, recursive=True)
        client.write('samples/' + sample + '/' + metrics_file_name, csvfile)


def get_results(client, sample):
    results_all = dict()
    for i in range(1, 4):
        dir_path = 'subgraphs/' + sample + '/' + str(i)
        fnames = client.list(dir_path)
        results = dict()
        for fname in fnames:
            with client.read(dir_path + '/' + fname, encoding='utf-8') as reader:
                for line in reader:
                    parts = line.split(',', 1)
                    label_str = parts[0]
                    label_str = label_str[2:len(label_str) - 1].strip()
                    tuple_freq_list = parts[1].split(',', 1)
                    freq = int(tuple_freq_list[0][2:])
                    subgraphs_str = tuple_freq_list[1][0:len(tuple_freq_list[1]) - 3].strip()
                    subgraphs_list = ast.literal_eval(subgraphs_str)

                    if label_str not in results:
                        results[label_str] = dict()
                        results[label_str]['freq'] = freq
                        results[label_str]['graphs'] = subgraphs_list
                    else:
                        results[label_str]['freq'] += freq
                        results[label_str]['graphs'].extend(subgraphs_list)
            results_all[i] = results
    return results_all


def check_edges(graph, graphs_br):
    res_list = []
    graph_set = set(graph[1])
    for index in range(graph[0] + 1, len(graphs_br)):
        graph_br_set = set(graphs_br[index])
        intersection = graph_set.intersection(graph_br_set)
        if 0 < len(intersection):
            res_list.append((graph[0], index))
    return res_list


def add_edge(edge, nx_g):
    nx_g.add_edge(edge[0], edge[1])


client = Config().get_client('dev')
samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367', 'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
conf = SparkConf().setAppName('FrequencyFix').setMaster('local[*]')
sc = SparkContext(conf=conf)
for sample in samples:
    results = get_results(client, sample)
    # build an overlap graph with triangles
    for size in results:
        if size == 1:
            continue
        for pattern in results[size]:
            graphs = results[size][pattern]['graphs']
            indexed_graphs = []
            index = 0
            for graph in graphs:
                indexed_graphs.append((index, graph))
                index += 1
            graphs_rdd = sc.parallelize(indexed_graphs)
            # print graphs_rdd.first()
            graphs_br = sc.broadcast(graphs)
            mapped_edges = graphs_rdd.flatMap(lambda graph: check_edges(graph, graphs_br.value))
            if not mapped_edges.isEmpty():
                nx_g = nx.Graph()
                nx_g.add_edges_from(mapped_edges.collect())
                graphs_connected = list(nx.connected_component_subgraphs(nx_g))
                freq = 0
                for graph in graphs_connected:
                    freq += len(maximum_independent_set(graph))
                print 'FREQ:' + str(freq)