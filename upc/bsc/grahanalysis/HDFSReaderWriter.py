from hdfs import Config
import ast
import csv
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set, max_clique
from networkx.algorithms import maximal_independent_set
import igraph as ig
import matplotlib.pyplot as plt



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

client = Config().get_client('dev')
samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367', 'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
for sample in samples:
    if sample != '7d734d06-f2b1-4924-a201-620ac8084c49':
        continue
    results = get_results(client, sample)
    # build an overlap graph with triangles
    test_graphs = results[3]['0:  1 2;  1:  0 2;  2:  0 1;']['graphs']
    nodes = range(len(test_graphs))
    nx_g = nx.Graph()
    i_g = ig.Graph()
    nx_g.add_nodes_from(nodes)
    i_g.add_vertices(len(test_graphs))
    edges_set = set()
    for index0 in nodes:
        for index1 in nodes:
            if index0 < index1:
                # if two graphs under indexes have at least one edge in common - draw an edge
                graph0 = set(test_graphs[index0])
                graph1 = set(test_graphs[index1])
                intersection = graph0.intersection(graph1)
                if 0 < len(intersection):
                    # build a edge
                    if (index0, index1) not in edges_set:
                        edges_set.add((index0, index1))
                        # print intersection
                        nx_g.add_edge(index0, index1)
                        i_g.add_edge(index0, index1)
    graphs = list(nx.connected_component_subgraphs(nx_g))
    print len(graphs)
    for graph in graphs:
        # nx.draw(graph)
        # plt.show()
        print 'MAXIMUM ' + str(len(maximum_independent_set(graph)))
        print 'MAXIMAL ' + str(len(maximal_independent_set(graph)))
        graph_compl = nx.complement(graph)
        print 'MAXIMUM CLIQUE ' + str(len(max_clique(graph_compl)))
    # print len(maximum_independent_set(nx_g))
    # print i_g.independence_number()




