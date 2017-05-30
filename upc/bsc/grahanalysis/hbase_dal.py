import happybase
import pyhive

from hdfs import Config
import ast
import networkx as nx
from pynauty import *
import os
from os import walk

CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]

data = None

pool = happybase.ConnectionPool(size=3, host='localhost')



def generating_parent(c_child, c_parent):
    # find all edges of the child
    # '  0:  0;  1:  6 7;  2:  2 8;  3:  3 13;  4:  4 12;  5:  5 9 13;  6:  1 6 11;  7:  1 7 14;  8:  2 8 9;  9:  5 8 10 14; 10:  9 12 14 15; 11:  6 11 13 15; 12:  4 10 12 15; 13:  3 5 11 13; 14:  7 9 10 14 15; 15:  10 11 12 14 15;'
    edges = list()
    adj_elements = c_child[:-1].split(';')
    for adj_line in adj_elements:
        adj_line = adj_line.strip()
        edges_line = adj_line.split(':')
        out_edge = int(edges_line[0])
        in_edges = edges_line[1].strip().split(' ')
        for element in in_edges:
            in_edge = int(element)
            if (out_edge < in_edge) and ((out_edge, in_edge) not in edges):
                edges.append((out_edge, in_edge))
            else:
                if (in_edge, out_edge) not in edges:
                    edges.append((in_edge, out_edge))

    # deleting last edge
    orig_edges = list(edges)
    last_index = len(edges) - 1
    last_edge = orig_edges[last_index]
    while last_edge is not None:
        edges = list(orig_edges)
        edges.remove(last_edge)
        # create networkx graph
        nx_g = nx.Graph()
        nx_g.add_edges_from(edges)
        if not nx.is_connected(nx_g):
            print 'Graph is not connected'
            last_index -= 1
            print 'Last index is ' + str(last_index)
            last_edge = orig_edges[last_index]
        else:
            last_edge = None
    # found good graph, now create the pynauty version
    vertices = set()
    for edge in edges:
        vertices.add(edge[0])
        vertices.add(edge[1])
    vertices_list = list(vertices)
    g = Graph(len(vertices_list))
    for edge in edges:
        g.connect_vertex(vertices_list.index(edge[0]), vertices_list.index(edge[1]))
    label = canon_label(g)
    return label.strip() == c_parent


def chr_str(chr_index):
    chrom_str = 'chr'
    if chr_index == 22:
        chrom_str += 'X'
    else:
        if chr_index == 23:
            chrom_str += 'Y'
        else:
            chrom_str += str(chr_index + 1)
    return chrom_str


def find_relative_position(position):
    # index = chr_number(chromosome)
    offset = 0L
    position = float(position)
    for i in range(len(CHR_MAP)):
        if offset < position < offset + long(CHR_MAP[i]):
            chrom_str = chr_str(i)
            pos = position - offset
            return chrom_str, pos
        else:
            offset += long(CHR_MAP[i])


def return_samples_metrics():
    #client = Config().get_client('dev')
    results_all = dict()
    samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
               'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
    settings_dir = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
    METRICS_FOLDER = os.path.join(PROJECT_ROOT, 'textparser/static/textparser/metrics/')
    for sample in samples:
        result_in_sample = dict()
        dir_path = METRICS_FOLDER + sample + '/'
        results = dict()
        fnames = []
        for (dirpath, dirnames, filenames) in walk(dir_path):
            fnames.extend(filenames)
            for fname in fnames:
                with open(dir_path + fname) as f:
                    content = f.readlines()
                    for line in content:
                        parts = line.split(',')
                        metric_str = parts[0]
                        val = parts[1]
                        if metric_str != 'metric':
                            result_in_sample[metric_str] = val
        results_all[sample] = result_in_sample
        # print results_all
    return results_all


def build_tree(parent_label, child_index, imax, res, current_dict):
    if child_index == imax:
        # nodes
        current_dict[parent_label] = list()
        for child_label in res[child_index]:
            if generating_parent(child_label, parent_label):
                current_dict[parent_label].append(child_label)
        return current_dict[parent_label]
    else:
        current_dict[parent_label] = dict()
        for child_label in res[child_index]:
            if generating_parent(child_label, parent_label):
                current_dict[parent_label][child_label] = build_tree(child_label, child_index + 1, imax, res, current_dict[parent_label])
        return current_dict


def get_pattern_view(bandwidth, threshold_counter):

    results_all = dict()
    samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
               'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
    dict_per_pattern = dict()
    with pool.connection() as connection:
        sample_pattern_table = connection.table('sample_pattern')
        for key, data in sample_pattern_table.scan(row_prefix=('b' + str(bandwidth) + 't' + str(threshold_counter)).encode('utf-8')):
            sample = data[b's:name']
            cancer = data[b's:cancer']
            bandwidth = data[b'b:value']
            threshold_counter = data[b't:counter']
            threshold = data[b't:value']
            pattern = data[b'p:code']
            parent = data[b'p:parent']
            size = int(data[b'p:size'])
            # list of tuples of strings
            inter_embeddings = [(inter_key, inter_value) for inter_key, inter_value in data.iteritems() if inter_key.startswith(b'e:inter_')]
            inter_edges = [map(float, val[0].split('_')[1].split(':')) for val in inter_embeddings]
            #print inter_edges
            intra_embeddings = [(intra_key, intra_value) for intra_key, intra_value in data.iteritems() if intra_key.startswith(b'e:intra_')]
            intra_edges = [map(float, val[0].split('_')[1].split(':')) for val in intra_embeddings]
            #print intra_edges
            if pattern not in dict_per_pattern:
                dict_per_pattern[pattern] = {'pattern': pattern, 'parent': parent, 'size': size, 'samples': [(sample, cancer)]}
            else:
                dict_per_pattern[pattern]['samples'].append((sample, cancer))

    # create hierarchy
    list_per_pattern = [value for value in dict_per_pattern.itervalues()]
    return list_per_pattern
    # for sample in samples:
    #     result_in_sample = dict()
    #     for i in range(1, 4):
    #         dir_path = 'subgraphs/' + sample + '/' + str(i)
    #         fnames = client.list(dir_path)
    #         results = dict()
    #         for fname in fnames:
    #             with client.read(dir_path + '/' + fname, encoding='utf-8') as reader:
    #                 for line in reader:
    #                     parts = line.split(',', 1)
    #                     label_str = parts[0]
    #                     label_str = label_str[2:len(label_str) - 1].strip()
    #                     tail_str = parts[1].strip()
    #                     tail = ast.literal_eval(tail_str[0:len(tail_str) - 1])
    #                     # print subgraphs_list
    #                     subgraphs_list_for_circos = list()
    #                     edges_set = set()
    #                     for subgraph_inter in tail[0][1]:
    #                         # print subgraph
    #                         for edge in subgraph_inter:
    #                             if edge not in edges_set:
    #                                 edges_set.add(edge)
    #                                 chrom1, pos1 = find_relative_position(edge[0])
    #                                 chrom2, pos2 = find_relative_position(edge[1])
    #                                 candidate_edge = {'source_id': chrom1,
    #                                                   'source_breakpoint': str(pos1),
    #                                                   'target_id': chrom2,
    #                                                   'target_breakpoint': str(pos2),
    #                                                   'source_label': 'inter',
    #                                                   'target_label': 'inter'}
    #                                 subgraphs_list_for_circos.append(candidate_edge)
    #                     for subgraph_intra in tail[1][1]:
    #                         # print subgraph
    #                         for edge in subgraph_intra:
    #                             if edge not in edges_set:
    #                                 edges_set.add(edge)
    #                                 chrom1, pos1 = find_relative_position(edge[0])
    #                                 chrom2, pos2 = find_relative_position(edge[1])
    #                                 candidate_edge = {'source_id': chrom1,
    #                                                   'source_breakpoint': str(pos1),
    #                                                   'target_id': chrom2,
    #                                                   'target_breakpoint': str(pos2),
    #                                                   'source_label': 'intra',
    #                                                   'target_label': 'intra'}
    #                                 subgraphs_list_for_circos.append(candidate_edge)
    #
    #                     if label_str not in results:
    #                         results[label_str] = dict()
    #                         results[label_str]['freq_inter'] = tail[0][0]
    #                         results[label_str]['graphs'] = subgraphs_list_for_circos
    #                         results[label_str]['freq_intra'] = tail[1][0]
    #                         # results[label_str]['graphs_intra'] = tail[1][1]
    #                     else:
    #                         results[label_str]['freq_inter'] += tail[0][0]
    #                         results[label_str]['graphs'].extend(subgraphs_list_for_circos)
    #                         results[label_str]['freq_intra'] += tail[1][0]
    #                         # results[label_str]['graphs_intra'].extend(tail[1][1])
    #             result_in_sample[i] = results
    #     results_all[sample] = result_in_sample
    #
    # return data


def build_tree_pattern_view(pattern, results):
    json_dict_local = {'text': pattern['pattern'], 'tags': [len(pattern['samples'])]}
    children = [val for val in results if val['parent'] == pattern['pattern']]
    if len(children) > 0:
        json_dict_local['nodes'] = []
        for child in children:
            json_dict_local['nodes'].append(build_tree_pattern_view(child, results))
    return json_dict_local


def get_hierarchical_pattern_view(bandwidth, threshold_counter, threshold=-1.0):
    results = get_pattern_view(bandwidth, threshold_counter)
    imax = max(results, key=lambda x: x['size'])['size']
    root_patterns = [val for val in results if val['parent'] == '0']
    json_dict = []
    for pattern in root_patterns:
        json_dict_local = (build_tree_pattern_view(pattern, results))
        json_dict.append(json_dict_local)
    return json_dict


def return_hierarchical_results(bandwidth, threshold_counter, threshold=-1.0):
    results = get_pattern_view(bandwidth, threshold_counter)
    res_per_sample = dict()
    freq_per_sample_pattern = dict()
    for sample in results.keys():
        last_level_dict = dict()
        i = 1
        imax = len(results[sample])
        for parent in results[sample][i]:
            last_level_dict = build_tree(parent, i + 1, imax, results[sample], last_level_dict)
        res_per_sample[sample] = last_level_dict
        freq_per_sample_pattern[sample] = {}
        for size in range(1, len(results[sample]) + 1):
            for pattern in results[sample][size]:
                freq_per_sample_pattern[sample][pattern] = [results[sample][size][pattern]['freq_inter'], results[sample][size][pattern]['freq_intra']]
    return freq_per_sample_pattern, res_per_sample


# def return_analysis_results_local():
#     if data is not None:
#         return data
#     settings_dir = os.path.dirname(__file__)
#     PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
#     SUBGRAPHS_FOLDER = os.path.join(PROJECT_ROOT, 'textparser/static/textparser/subgraphs/')
#     results_all = dict()
#     samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
#                'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
#     for sample in samples:
#         result_in_sample = dict()
#         for i in range(1, 4):
#             dir_path = SUBGRAPHS_FOLDER + sample + '/' + str(i) + '/'
#             results = dict()
#             fnames = []
#             for (dirpath, dirnames, filenames) in walk(dir_path):
#                 fnames.extend(filenames)
#                 for fname in fnames:
#                     with open(dir_path + fname) as f:
#                         content = f.readlines()
#                         for line in content:
#                             parts = line.split(',', 1)
#                             label_str = parts[0]
#                             label_str = label_str[2:len(label_str) - 1].strip()
#                             tail_str = parts[1].strip()
#                             tail = ast.literal_eval(tail_str[0:len(tail_str) - 1])
#                             # print subgraphs_list
#                             subgraphs_list_for_circos = list()
#                             edges_set = set()
#                             for subgraph_inter in tail[0][1]:
#                                 # print subgraph
#                                 for edge in subgraph_inter:
#                                     if edge not in edges_set:
#                                         edges_set.add(edge)
#                                         chrom1, pos1 = find_relative_position(edge[0])
#                                         chrom2, pos2 = find_relative_position(edge[1])
#                                         candidate_edge = {'source_id': chrom1,
#                                                           'source_breakpoint': str(pos1),
#                                                           'target_id': chrom2,
#                                                           'target_breakpoint': str(pos2),
#                                                           'source_label': 'inter',
#                                                           'target_label': 'inter'}
#                                         subgraphs_list_for_circos.append(candidate_edge)
#                             for subgraph_intra in tail[1][1]:
#                                 # print subgraph
#                                 for edge in subgraph_intra:
#                                     if edge not in edges_set:
#                                         edges_set.add(edge)
#                                         chrom1, pos1 = find_relative_position(edge[0])
#                                         chrom2, pos2 = find_relative_position(edge[1])
#                                         candidate_edge = {'source_id': chrom1,
#                                                           'source_breakpoint': str(pos1),
#                                                           'target_id': chrom2,
#                                                           'target_breakpoint': str(pos2),
#                                                           'source_label': 'intra',
#                                                           'target_label': 'intra'}
#                                         subgraphs_list_for_circos.append(candidate_edge)
#
#                             if label_str not in results:
#                                 results[label_str] = dict()
#                                 results[label_str]['freq_inter'] = tail[0][0]
#                                 results[label_str]['graphs'] = subgraphs_list_for_circos
#                                 results[label_str]['freq_intra'] = tail[1][0]
#                                 # results[label_str]['graphs_intra'] = tail[1][1]
#                             else:
#                                 results[label_str]['freq_inter'] += tail[0][0]
#                                 results[label_str]['graphs'].extend(subgraphs_list_for_circos)
#                                 results[label_str]['freq_intra'] += tail[1][0]
#                                 # results[label_str]['graphs_intra'].extend(tail[1][1])
#                     result_in_sample[i] = results
#         results_all[sample] = result_in_sample
#         global data
#         data = results_all
#     return data


# def return_analysis_results():
#     if data is not None:
#         return data
#     client = Config().get_client('dev')
#     results_all = dict()
#     samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
#                'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
#     for sample in samples:
#         result_in_sample = dict()
#         for i in range(1, 4):
#             dir_path = 'subgraphs/' + sample + '/' + str(i)
#             fnames = client.list(dir_path)
#             results = dict()
#             for fname in fnames:
#                 with client.read(dir_path + '/' + fname, encoding='utf-8') as reader:
#                     for line in reader:
#                         parts = line.split(',', 1)
#                         label_str = parts[0]
#                         label_str = label_str[2:len(label_str) - 1].strip()
#                         tail_str = parts[1].strip()
#                         tail = ast.literal_eval(tail_str[0:len(tail_str) - 1])
#                         # print subgraphs_list
#                         subgraphs_list_for_circos = list()
#                         edges_set = set()
#                         for subgraph_inter in tail[0][1]:
#                             # print subgraph
#                             for edge in subgraph_inter:
#                                 if edge not in edges_set:
#                                     edges_set.add(edge)
#                                     chrom1, pos1 = find_relative_position(edge[0])
#                                     chrom2, pos2 = find_relative_position(edge[1])
#                                     candidate_edge = {'source_id': chrom1,
#                                                       'source_breakpoint': str(pos1),
#                                                       'target_id': chrom2,
#                                                       'target_breakpoint': str(pos2),
#                                                       'source_label': 'inter',
#                                                       'target_label': 'inter'}
#                                     subgraphs_list_for_circos.append(candidate_edge)
#                         for subgraph_intra in tail[1][1]:
#                             # print subgraph
#                             for edge in subgraph_intra:
#                                 if edge not in edges_set:
#                                     edges_set.add(edge)
#                                     chrom1, pos1 = find_relative_position(edge[0])
#                                     chrom2, pos2 = find_relative_position(edge[1])
#                                     candidate_edge = {'source_id': chrom1,
#                                                       'source_breakpoint': str(pos1),
#                                                       'target_id': chrom2,
#                                                       'target_breakpoint': str(pos2),
#                                                       'source_label': 'intra',
#                                                       'target_label': 'intra'}
#                                     subgraphs_list_for_circos.append(candidate_edge)
#
#                         if label_str not in results:
#                             results[label_str] = dict()
#                             results[label_str]['freq_inter'] = tail[0][0]
#                             results[label_str]['graphs'] = subgraphs_list_for_circos
#                             results[label_str]['freq_intra'] = tail[1][0]
#                             # results[label_str]['graphs_intra'] = tail[1][1]
#                         else:
#                             results[label_str]['freq_inter'] += tail[0][0]
#                             results[label_str]['graphs'].extend(subgraphs_list_for_circos)
#                             results[label_str]['freq_intra'] += tail[1][0]
#                             # results[label_str]['graphs_intra'].extend(tail[1][1])
#                 result_in_sample[i] = results
#         results_all[sample] = result_in_sample
#         global data
#         data = results_all
#         # print results_all
#     return data

res = get_hierarchical_pattern_view(50000, 2)
print res



