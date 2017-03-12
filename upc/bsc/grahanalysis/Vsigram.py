from pynauty import *
import csv

from upc.bsc.grahanalysis.model.VsigramGraph import VsigramGraph


def map_csv_to_graph(path):
    with open('c:\\Users\Katherine\\Documents\\thesis\\data\\pattern_mining\\new_assignment_dir.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['pos_1', 'pos_2'])
        next(csvfile)
        positions = list()
        edges = list()
        for row in reader:
            positions.append((row['pos_1'], row['pos_2']))

        vertices_set = set()
        for position in positions:
            vertices_set.add(position[0])
            vertices_set.add(position[1])
        vertices_list = list(vertices_set)
        g = Graph(len(vertices_list))
        for edge in positions:
            if len(edge) > 0:
                g.connect_vertex(vertices_list.index(edge[0]), vertices_list.index(edge[1]))
                edges.append((vertices_list.index(edge[0]), vertices_list.index(edge[1])))
        # cert = certificate(g)
        return VsigramGraph(g, certificate(g), 1, 1, edges, vertices_list)


def vsigram(G, minFreq):
    # CLf = NULL //a set of frequent canonical labels
    CLf = set()
    # CLf1 = all frequent canonical labels of size 1 (edge) subgraphs in G
    CLf1 = set()
    MCLf = dict()
    MCLf1 = dict()
    for edge in G.edges:
        # ceate new graph - subgraph of
        vertices = set(edge)
        g = Graph(len(vertices))
        g.connect_vertex(edge[0], edge[1])
        label = certificate(g)
        subgraph1 = VsigramGraph(g, label, edges=list(edge), vertices=list(vertices))
        if subgraph1.label not in CLf1:
            CLf1.add(subgraph1.label)
            MCLf1[label] = list(subgraph1)
        else:
            MCLf1[label].append(subgraph1)

    # for each clf1 in CLf1 do
    for key in MCLf1.keys():
        # M(clf1) =  all subgraphs with canonical label clf1
        # remove an element from dictionary if number of subgraphs < minFreq
        if len(MCLf1[key]) < minFreq:
            del MCLf1[key]
    # end for

    # for each clf1 in CLf1 do
    for clf1 in CLf1:
        # CLf = CLf + vsigram_exten(clf1, G, minFreq)
        CLf.add(vsigram_extend(clf1, MCLf1[clf1], G, minFreq, MCLf, CLf))
    # end for
    # return CLf
    return CLf


def vsigram_extend(clfk, Mclfk, G, minFreq, MCLf, CLf):
    # Sk+1 = NULL
    Skplus1 = list()
    # CLk+1 = NULL  // stores new, distinct canonical labels
    CLkplus1 = set()
    MCLkplus1 = dict()
    # for each sk in M(clfk ) do
    for sk in Mclfk:
        # Sk+1 = Sk+1 + {sk + e | e from E} // extend subgraph with an edge
        for edge in G.edges:
            # check if edge is adjacent
            if edge[0] in sk.vertices or edge[1] in sk.vertices:
                # create new graph
                # new set of vertices
                vertices = set(sk.vertices)
                # add each vertex of the new edge (max one of them is added)
                vertices.add(edge[0])
                vertices.add(edge[1])
                # new list of edges from original edges
                edges = list(sk.edges)
                # add new edge
                edges.append(edge)
                g = Graph(len(vertices))
                for orig_edge in sk.edges:
                    g.connect_vertex(orig_edge[0], orig_edge[1])
                g.connect_vertex(edge[0], edge[1])
                label = certificate(g)
                subgraphkplus1 = VsigramGraph(g, label, edges=edges, vertices=list(vertices))
                Skplus1.append(subgraphkplus1)
    # end for
    # for each sk+1 in Sk+1
    for skplus1 in Skplus1:
        # CLk+1 = CLk+1 + (sk+1.label) // only distinct canonical labels
        CLkplus1.add(skplus1.label)
        # M(sk+1.label) = M(sk+1.label) + {sk+1} //add subgraph
        if skplus1.label not in MCLkplus1:
            MCLkplus1[skplus1.label] = list(skplus1)
        else:
            MCLkplus1[skplus1.label].append(skplus1)
    # end for
    # for each clk+1 in CLk+1 do
    for clkplus1 in CLkplus1:
        # if clfk is not the generating parent of clk+1 then
        if False: # generating parent function
            continue
        # end if
        # compute clk+1.freq from M(clk+1)
        # if clk+1.freq < minFreq then
        if len(MCLkplus1[clkplus1]) < minFreq:
            continue
        # end if
        # CLf = CLf + {clk+1} + vsigram_extend(clk+1, G, minFreq)
        CLf.add(clkplus1)
        CLf.add(clkplus1, G, minFreq, MCLf, CLf)
    # end for
    return CLf


graph = map_csv_to_graph('c:\\Users\\Katherine\\Documents\\thesis\\data\\pattern_mining\\new_assignment_dir.csv')
frequent_subgraphs = vsigram(graph, 1)
