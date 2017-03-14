from pynauty import *
import csv

from upc.bsc.grahanalysis.model.VsigramGraph import VsigramGraph


def map_csv_to_graph(path='/home/kkrasnas/Documents/thesis/pattern_mining/new_assignment.csv'):
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['pos_1', 'pos_2'])
        next(csvfile)
        positions = list()
        edges_set = set()
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
            # always minIndex -> maxIndex to avoid duplicate edges
            if vertices_list.index(edge[0]) < vertices_list.index(edge[1]):
                g.connect_vertex(vertices_list.index(edge[0]), vertices_list.index(edge[1]))
                edges_set.add((vertices_list.index(edge[0]), vertices_list.index(edge[1])))
            else:
                g.connect_vertex(vertices_list.index(edge[1]), vertices_list.index(edge[0]))
                edges_set.add((vertices_list.index(edge[1]), vertices_list.index(edge[0])))
        edges = list(edges_set)
        return VsigramGraph(g, certificate(g), canon_label(g), 1, 1,
                            edges=edges, vertices=range(len(vertices_list)),
                            orig_edges=positions, orig_vertices=vertices_list)


def vsigram(G, minFreq):
    # CLf = NULL //a set of frequent canonical labels
    CLf = set()
    # CLf1 = all frequent canonical labels of size 1 (edge) subgraphs in G
    CLf1 = set()
    labels1 = set()
    MCLf = dict()
    MCLf1 = dict()
    for edge in G.edges:
        # ceate new graph - subgraph of
        vertices = set(edge)
        g = Graph(len(vertices))
        vertices_list = list(vertices)
        g.connect_vertex(vertices_list.index(edge[0]), vertices_list.index(edge[1]))
        edges_list = list()
        edges_list.append((vertices_list.index(edge[0]), vertices_list.index(edge[1])))
        label = certificate(g)
        #aut = autgrp(g)
        #print label.decode('utf-8')
        subgraph1 = VsigramGraph(g, label, canon_label(g),
                                 edges=edges_list, vertices=range(len(vertices_list)),
                                 orig_edges=[edge], orig_vertices=vertices_list)
        if subgraph1.label not in CLf1:
            CLf1.add(subgraph1.label)
            MCLf1[label] = [subgraph1]
        else:
            MCLf1[label].append(subgraph1)
        labels1.add(subgraph1.label_arr)

    # for each clf1 in CLf1 do
    for key in MCLf1.keys():
        # M(clf1) =  all subgraphs with canonical label clf1
        # remove an element from dictionary if number of subgraphs < minFreq
        if len(MCLf1[key]) < minFreq:
            #remove from frequent labels and frequent subgraphs
            del MCLf1[key]
            CLf1.remove(key)
        else:
            # add to all frequent
            CLf.add(key)
            MCLf[key] = MCLf1[key]

    # end for

    # for each clf1 in CLf1 do
    for clf1 in CLf1:
        # CLf = CLf + vsigram_exten(clf1, G, minFreq)
        CLf.update(vsigram_extend(clf1, MCLf1[clf1], G, minFreq, MCLf, CLf, 1+1))
    # end for
    # return CLf
    return CLf


def vsigram_extend(clfk, Mclfk, G, minFreq, MCLf, CLf, size):
    print 'Processing size ' + str(size)
    # Sk+1 = NULL
    Skplus1 = list()
    # CLk+1 = NULL  // stores new, distinct canonical labels
    CLkplus1 = set()
    labelsplus1 = set()
    MCLkplus1 = dict()
    # for each sk in M(clfk ) do
    for sk in Mclfk:
        # Sk+1 = Sk+1 + {sk + e | e from E} // extend subgraph with an edge
        for orig_edge in G.edges:
            # check if edge is adjacent
            if (orig_edge not in sk.orig_edges) \
                    and (orig_edge[0] in sk.orig_vertices or orig_edge[1] in sk.orig_vertices):
                # create new graph
                # new set of vertices
                orig_vertices = set(sk.orig_vertices)
                # add each vertex of the new edge (max one of them is added)
                orig_vertices.add(orig_edge[0])
                orig_vertices.add(orig_edge[1])
                orig_vertices_list = list(orig_vertices)
                # new list of edges from original edges
                orig_edges = list(sk.orig_edges)
                # add new edge
                orig_edges.append(orig_edge)
                g = Graph(len(orig_vertices))
                edges = list()
                for new_orig_edge in orig_edges:
                    g.connect_vertex(orig_vertices_list.index(new_orig_edge[0]), orig_vertices_list.index(new_orig_edge[1]))
                    edges.append((orig_vertices_list.index(new_orig_edge[0]), orig_vertices_list.index(new_orig_edge[1])))
                label = certificate(g)
                subgraphkplus1 = VsigramGraph(g, label, canon_label(g),
                                              edges=edges, vertices=range(len(orig_vertices_list)),
                                              orig_edges=orig_edges, orig_vertices=orig_vertices_list)
                Skplus1.append(subgraphkplus1)
    # end for
    # for each sk+1 in Sk+1
    for skplus1 in Skplus1:
        # CLk+1 = CLk+1 + (sk+1.label) // only distinct canonical labels
        CLkplus1.add(skplus1.label)
        labelsplus1.add(skplus1.label_arr)
        # M(sk+1.label) = M(sk+1.label) + {sk+1} //add subgraph
        if skplus1.label not in MCLkplus1:
            MCLkplus1[skplus1.label] = [skplus1]
        else:
            MCLkplus1[skplus1.label].append(skplus1)
    # end for
    # for each clk+1 in CLk+1 do
    for clkplus1 in CLkplus1:
        # if clfk is not the generating parent of clk+1 then
        if False: # !generating_parent():  # generating parent function
            continue
        # end if
        # compute clk+1.freq from M(clk+1)
        # if clk+1.freq < minFreq then
        if len(MCLkplus1[clkplus1]) < minFreq:
            continue
        # end if
        # CLf = CLf + {clk+1} + vsigram_extend(clk+1, G, minFreq)
        CLf.add(clkplus1)
        if(clkplus1 not in MCLf):
            MCLf[clkplus1] = MCLkplus1[clkplus1]
        else:
            MCLf[clkplus1].append(MCLkplus1[clkplus1])
        CLf.update(vsigram_extend(clkplus1, MCLkplus1[clkplus1], G, minFreq, MCLf, CLf, size+1))
    # end for
    return CLf


graph = map_csv_to_graph()
frequent_subgraphs = vsigram(graph, 1)
