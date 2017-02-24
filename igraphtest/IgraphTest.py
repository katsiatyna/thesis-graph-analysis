import csv
import re
import igraph



with open('/home/katsiaryna/Documents/thesis/pattern_mining/tables/rules_sample.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['1:nrow(p3)', 'numprecs', 'rules', 'support', 'confidence', 'lift',
                                                 'precedent', 'consequent', 'code'])
    freq_itemsets = []
    precedent_graps = []
    consequent_graphs = []
    groups = set()
    for row in reader:
        #if row['1:nrow(p3)'] in ['13969079', '13969099', '13969119', '13969139', '13969159', '13969179', '13969199']:
        if row['1:nrow(p3)'] in ['3699499','3699519','3699539','3699559','3699579']:
            #consequent graph
            p = re.compile('\{(.*)\}')
            conseq = p.search(row['consequent']).group(1)
            vertices = conseq.split('.')
            full_vertices = conseq.split('.')
            vertices_set = set(vertices)
            full_set = set(full_vertices)
            full_graph = igraph.Graph()
            conseq_graph = igraph.Graph()

            conseq_graph.add_vertices(list(vertices_set))
            conseq_graph.add_edges([(vertices[0], vertices[1])])

            consequent_graphs.append(conseq_graph)
            #antecedent graph
            ante = p.search(row['precedent']).group(1)
            edges = ante.split(',')
            ante_graph = igraph.Graph()
            vertices_set = set()
            for edge in edges:
                vertices = edge.split('.')
                vertices_set |= set(vertices)
                full_set |= set(vertices)
            ante_graph.add_vertices(list(vertices_set))
            full_graph.add_vertices(sorted(list(full_set)))
            full_vertices = sorted(full_vertices)
            full_graph.add_edges([(full_vertices[0], full_vertices[1])])
            sorted_edges = []
            for edge in edges:
                vertices = edge.split('.')
                if len(vertices) > 1:
                    ante_graph.add_edges([(vertices[0], vertices[1])])
                    #vertices = sorted(vertices)
                    full_graph.add_edges([(vertices[0], vertices[1])])

            precedent_graps.append(ante_graph)
            freq_itemsets.append(full_graph)
            groups.add(str(full_graph.canonical_permutation()))
            #print(row['1:nrow(p3)'], full_graph.canonical_permutation())
            #print(full_graph)
    #print groups
    for graph1 in freq_itemsets:
        for graph2 in freq_itemsets:
            print (graph1.isomorphic(graph2), graph1.isomorphic_bliss(graph2), graph1.isomorphic_vf2(graph2),
                   graph1.canonical_permutation(), graph2.canonical_permutation())

