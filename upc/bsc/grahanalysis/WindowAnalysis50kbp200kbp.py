import csv
import networkx as nx
import igraph as ig
from networkx.algorithms.approximation import maximum_independent_set
from networkx.algorithms import maximal_independent_set


def chr_number(chr_str):
    if chr_str == 'X':
        index = 22
    else:
        if chr_str == 'Y':
            index = 23
        else:
            index = int(chr_str) - 1
    return index


def find_absolute_position(position, chromosome):
    index = chr_number(chromosome)
    offset = 0L
    if index != 0:
        for i in range(index):
            offset += long(CHR_MAP[i])
    return offset + long(position)


# info chromosome
CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]

# read the file from Luisa and analyze the triangles
with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/7d734d06-f2b1-4924-a201-620ac8084c49.chromplex.wsinter50000.wsintra200000.csv', 'rb') as chromoplexia_csv:
    reader = csv.DictReader(chromoplexia_csv, delimiter= '\t',fieldnames=['pos_1', 'pos_2', 'pos_3'])
    # ('6', '6', 42536734, 44191141)	('6', '12', 44191196, 76701650)	('12', '6', 76701958, 42536907)
    chromoplexies = list()
    i = 0
    for line in reader:
        # read each edge
        str1 = line['pos_1'][1:len(line['pos_1']) - 1]
        edge_list = str1.split(',')
        # print edge1_list
        vertex1 = find_absolute_position(long(edge_list[2].strip()), edge_list[0].strip()[1:len(edge_list[0].strip()) - 1])
        vertex2 = find_absolute_position(long(edge_list[3].strip()), edge_list[1].strip()[1:len(edge_list[1].strip()) - 1])
        if vertex1 < vertex2:
            edge1 = (vertex1, vertex2)
        else:
            edge1 = (vertex2, vertex1)
        str2 = line['pos_2'][1:len(line['pos_2']) - 1]
        edge_list = str2.split(',')
        # print edge1_list
        vertex1 = find_absolute_position(long(edge_list[2].strip()), edge_list[0].strip()[1:len(edge_list[0].strip()) - 1])
        vertex2 = find_absolute_position(long(edge_list[3].strip()), edge_list[1].strip()[1:len(edge_list[1].strip()) - 1])
        if vertex1 < vertex2:
            edge2 = (vertex1, vertex2)
        else:
            edge2 = (vertex2, vertex1)
        str3 = line['pos_3'][1:len(line['pos_3']) - 1]
        edge_list = str3.split(',')
        # print edge1_list
        vertex1 = find_absolute_position(long(edge_list[2].strip()), edge_list[0].strip()[1:len(edge_list[0].strip()) - 1])
        vertex2 = find_absolute_position(long(edge_list[3].strip()), edge_list[1].strip()[1:len(edge_list[1].strip()) - 1])
        if vertex1 < vertex2:
            edge3 = (vertex1, vertex2)
        else:
            edge3 = (vertex2, vertex1)
        chromoplexy = (i, set([edge1, edge2, edge3]))
        i += 1
        chromoplexies.append(chromoplexy)

# build overlap graph
nx_g = nx.Graph()
i_g = ig.Graph()
nx_g.add_nodes_from(range(len(chromoplexies)))
i_g.add_vertices(len(chromoplexies))
edges_set = set()
for index0 in range(len(chromoplexies)):
    for index1 in range(index0 + 1, len(chromoplexies)):
        # if two graphs under indexes have at least one edge in common - draw an edge
        if index1 == index0 + 1 and index0 % 100 == 0:
            print index0
        graph0 = set(chromoplexies[index0][1])
        graph1 = set(chromoplexies[index1][1])
        intersection = graph0.intersection(graph1)
        if 0 < len(intersection):
            # build a edge
            if (index0, index1) not in edges_set:
                edges_set.add((index0, index1))
                # print intersection
                nx_g.add_edge(index0, index1)
                i_g.add_edge(index0, index1)
graphs = list(nx.connected_component_subgraphs(nx_g))
graphs_ig = i_g.components().subgraphs()
print 'Connected components NX: ' + str(len(graphs))
print 'Connected components IG: ' + str(len(graphs_ig))
for graph in graphs:
    # nx.draw(graph)
    # plt.show()
    print len(maximum_independent_set(graph))
    print len(maximal_independent_set(graph))
print '-------------------------'
for graph in graphs_ig:
    print graph.alpha()


# find common edges (2)
# print len(chromoplexies)
# mult = dict()
# for triangle in chromoplexies:
#     for candidate in chromoplexies:
#         if triangle[0] != candidate[0]:
#             intersection = triangle[1].intersection(candidate[1])
#             if 1 < len(intersection):
#                 # two edges in common
#                 intersection_hash = ''
#                 for tup in intersection:
#                     intersection_hash += str(tup[0]) + '.' + str(tup[1]) + ';'
#                 if intersection_hash in mult:
#                     if triangle not in mult[intersection_hash]:
#                         mult[intersection_hash].append(triangle)
#                     if candidate not in mult[intersection_hash]:
#                         mult[intersection_hash].append(candidate)
#                 else:
#                     mult[intersection_hash] = [triangle, candidate]
# with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/chromoplexies_analysis_50_200.csv', 'wb') as csvfile:
#         fieldnames = ['common edges', 'duplicate count (graphs - 1)', 'graphs']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for key in mult.keys():
#             writer.writerow({'common edges': key, 'duplicate count (graphs - 1)': len(mult[key]) - 1, 'graphs': mult[key]})
#
# # find all the unduplicated ones
# duplicates = list()
# chromoplexies_unduplicated = list()
# # pick a representative from each group
# for key in mult.keys():
#     i = 0
#     el = mult[key][i]
#     while (el in chromoplexies_unduplicated or el in duplicates) and i < len(mult[key]):
#         el = mult[key][i]
#         i += 1
#     if el not in chromoplexies_unduplicated:
#         chromoplexies_unduplicated.append(el)
#     # add duplicates
#     for item in mult[key]:
#         if item != el:
#             duplicates.append(item)
# for item in chromoplexies:
#     if item not in chromoplexies_unduplicated and item not in duplicates:
#         chromoplexies_unduplicated.append(item)
# print len(chromoplexies_unduplicated)
# chromoplexies_unduplicated = sorted(chromoplexies_unduplicated)
# for chrom in chromoplexies_unduplicated:
#     print chrom
#
# mult_check = dict()
# for triangle in chromoplexies_unduplicated:
#     for candidate in chromoplexies_unduplicated:
#         if triangle[0] != candidate[0]:
#             intersection = triangle[1].intersection(candidate[1])
#             if 1 < len(intersection):
#                 # two edges in common
#                 intersection_hash = ''
#                 for tup in intersection:
#                     intersection_hash += str(tup[0]) + '.' + str(tup[1]) + ';'
#                 if intersection_hash in mult_check:
#                     if triangle not in mult_check[intersection_hash]:
#                         mult_check[intersection_hash].append(triangle)
#                     if candidate not in mult_check[intersection_hash]:
#                         mult_check[intersection_hash].append(candidate)
#                 else:
#                     mult_check[intersection_hash] = [triangle, candidate]
# for key in mult_check:
#     print key + ': ' + str(len(mult_check[key]) - 1)
