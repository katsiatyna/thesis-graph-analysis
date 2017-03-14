import networkx as nx
import csv
import matplotlib.pyplot as plt

G=nx.Graph()

with open('/home/kkrasnas/Documents/thesis/pattern_mining/new_assignment.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['pos_1', 'pos_2'])
    next(csvfile)
    positions = []
    for row in reader:
        positions.append((row['pos_1'], row['pos_2']))

    vertices_set = set()
    for position in positions:
        vertices_set.add(position[0])
        vertices_set.add(position[1])
    vertices_list = list(vertices_set)
    # G.add_edge(position[0], position[1])
    G.add_nodes_from(range(len(vertices_list)))
    for edge in positions:
        G.add_edge(vertices_list.index(edge[0]), vertices_list.index(edge[1]))
    print G.number_of_edges()

    plt.figure(figsize=(10,10))
    nx.draw_circular(G, with_labels=False)
    plt.show()
    triangles = nx.triangles(G)
    print len(triangles)
    for key in triangles.keys():
        if triangles[key] > 0:
            print str(key) + ': ' + str(triangles[key])
