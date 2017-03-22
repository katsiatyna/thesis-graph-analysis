import networkx as nx
import csv
import pylab as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout, pygraphviz_layout


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
    print G.nodes_with_selfloops()


    plt.figure(figsize=(10,10))
    # nx.draw_circular(G, with_labels=True)
    # nx.draw_graphviz(G, prog="circo")
    nx.draw(G, pos=pygraphviz_layout(G, prog='circo'), node_size=1600, cmap=plt.cm.Blues,
         node_color=range(len(G)), with_labels=True)
    plt.gca().set_aspect('equal')
    plt.show()
    write_dot(G,'graph.dot')
    triangles = nx.triangles(G)
    print len(triangles)
    for key in triangles.keys():
        if triangles[key] > 0:
            print str(key) + ': ' + str(triangles[key])
