
from pynauty import *
import csv
import re

with open('/home/kkrasnas/Documents/thesis/pattern_mining/tables/rules_sample.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['1:nrow(p3)', 'numprecs', 'rules', 'support', 'confidence', 'lift',
                                                 'precedent', 'consequent', 'code'])
    freq_itemsets = []
    precedent_graps = []
    consequent_graphs = []
    groups = set()
    for row in reader:
        #if row['1:nrow(p3)'] in ['13969079', '13969099', '13969119', '13969139', '13969159', '13969179', '13969199']:
        #if row['1:nrow(p3)'] in ['3699499','3699519','3699539','3699559','3699579']:
            p = re.compile('\{(.*)\}')
            conseq = p.search(row['consequent']).group(1)
            full_vertices = conseq.split('.')
            full_set = set(map(int, full_vertices))
            full_edges = list()
            full_edges.append(sorted(map(int, full_vertices)))


            ante = p.search(row['precedent']).group(1)
            edges = ante.split(',')
            vertices_set = set()
            for edge in edges:
                vertices = edge.split('.')
                if len(vertices) > 1:
                    full_set |= set(map(int, vertices))
                    full_edges.append(sorted(map(int, vertices)))
            full_set = sorted(list(full_set))
            g = Graph(len(full_set))
            for edge in full_edges:
               g.connect_vertex(full_set.index(edge[0]), full_set.index(edge[1]))
            cert = certificate(g)
            freq_itemsets.append(cert)
            print (row['1:nrow(p3)'], full_set, cert)

    #for g in freq_itemsets:
     #   print('\n')
      #  for g1 in freq_itemsets:
       #     print(g == g1)


