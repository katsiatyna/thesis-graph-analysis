import happybase
import numpy as np
from matplotlib import pylab as plt
from matplotlib import pyplot
import peakutils
import statsmodels.api as sm
import csv
import sys
from upc.bsc.Constants import SAMPLES, CHR_MAP, BANDWIDTH_CANDIDATES, SAMPLE_CANCER
from operator import itemgetter
import os
import errno
from hdfs import Config
import ast
import networkx as nx

def write_init_files(client, sample, dir_path, bandwidth, threshold_counter):
    # write the assignment file
    file_name = sample + '_new_assignment.csv'
    metrics_file_name = sample + '_metrics.csv'
    with open(dir_path + file_name,
              'rw') as csvfile:
        client.delete('samples/' + sample + '/' + file_name, recursive=True)
        client.delete('samples/b' + str(int(bandwidth)) + '/' + sample + '/t' + str(threshold_counter) + '/' + file_name, recursive=True)
        client.write('samples/b' + str(int(bandwidth)) + '/' + sample + '/t' + str(threshold_counter) + '/' + file_name, csvfile)
    with open(dir_path + metrics_file_name,
              'rw') as csvfile:
        client.delete('samples/' + sample + '/' + metrics_file_name, recursive=True)
        client.delete('samples/b' + str(int(bandwidth)) + '/' + sample + '/t' + str(threshold_counter) + '/' + metrics_file_name, recursive=True)
        client.write('samples/b' + str(int(bandwidth)) + '/' + sample + '/t' + str(threshold_counter) + '/' + metrics_file_name, csvfile)


def chr_number(chr_str):
    if chr_str == 'X':
        chr_index = 22
    else:
        if chr_str == 'Y':
            chr_index = 23
        else:
            chr_index = int(chr_str) - 1
    return chr_index


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


def find_absolute_position(position, chromosome):
    index = chr_number(chromosome)
    offset = 0L
    if index != 0:
        for i in range(index):
            offset += long(CHR_MAP[i])
    return offset + long(position)


def find_relative_position(position, chrom_as_str=True):
    # index = chr_number(chromosome)
    offset = 0L
    for i in range(len(CHR_MAP) + 1):
        if offset < position < offset + long(CHR_MAP[i]):
            chrom_str = chr_str(i) if chrom_as_str else i
            pos = position - offset
            return chrom_str, pos
        else:
            offset += long(CHR_MAP[i])


def load_edges_2d(path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/7d734d06-f2b1-4924-a201-620ac8084c49_positions.csv'):
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['Chr_BKP_1', 'Pos_BKP_1', 'Chr_BKP_2', 'Pos_BKP_2'])
        next(csvfile)
        positions = []
        for row in reader:
            positions.append(((chr_number(row['Chr_BKP_1']),
                               find_absolute_position(row['Pos_BKP_1'], row['Chr_BKP_1'])),
                              (chr_number(row['Chr_BKP_2']),
                               find_absolute_position(row['Pos_BKP_2'], row['Chr_BKP_2']))))
        return positions


def convert_to_2d_array(edges):
    positions = dict()
    for edge in edges:
        if edge[0][0] not in positions.keys():
            positions[edge[0][0]] = {edge[0][1]}
        else:
            positions[edge[0][0]].add(edge[0][1])
        if edge[1][0] not in positions.keys():
            positions[edge[1][0]] = {edge[1][1]}
        else:
            positions[edge[1][0]].add(edge[1][1])
    for key in positions.keys():
        positions[key] = sorted(map(float, positions[key]))
    return positions


def func(x, return_val):
    return return_val


def write_undirect_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv'):

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'w+') as csvfile:
        fieldnames = ['pos_1', 'pos_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in assignment:
            writer.writerow({'pos_1': row[0], 'pos_2': row[1]})


def write_circos_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_for_circos.csv'):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # write new assignment
    with open(path, 'w+') as csvfile:
        fieldnames = ['source_id','source_breakpoint','target_id','target_breakpoint','source_label','target_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # print 'New assignment before deduplication: ' + str(len(assignment))
        assignment_deduplicated = []
        edges_set = set()
        for row in assignment:
            if row[0] < row[1]:
                candidate_edge = (row[0], row[1])
            else:
                candidate_edge = (row[1], row[0])
            if candidate_edge not in edges_set:
                edges_set.add(candidate_edge)
                assignment_deduplicated.append(candidate_edge)
        # print 'New assignment after deduplication: ' + str(len(assignment_deduplicated))
        writer.writeheader()
        for row in assignment_deduplicated:
            source_chr, source_rel_pos = find_relative_position(row[0])
            target_chr, target_rel_pos = find_relative_position(row[1])
            writer.writerow({'source_id': source_chr, 'source_breakpoint': source_rel_pos,
                             'target_id':target_chr, 'target_breakpoint': target_rel_pos,
                             'source_label':'', 'target_label':''})


def write_circos_input_file_orig(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_orig_for_circos.csv'):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # write new assignment
    with open(path, 'w+') as csvfile:
        fieldnames = ['source_id','source_breakpoint','target_id','target_breakpoint','source_label','target_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in assignment:
            source_chr, source_rel_pos = find_relative_position(row[0][1])
            target_chr, target_rel_pos = find_relative_position(row[1][1])
            writer.writerow({'source_id': source_chr, 'source_breakpoint': source_rel_pos,
                             'target_id':target_chr, 'target_breakpoint': target_rel_pos,
                             'source_label':'', 'target_label':''})


def diff(pos, peak):
    return abs(pos - peak)


def find_closest_peak_fft(pos, support_pos, peak_indexes):
    min_diff = sys.maxint
    new_pos = pos
    peak_ind = -1
    for peak in peak_indexes:
        if(diff(pos, support_pos[peak]) < min_diff):
            min_diff = diff(pos, support_pos[peak])
            new_pos = support_pos[peak]
            peak_ind = peak
    return new_pos, peak_ind


def construct_new_assignment_fft(original_pos, support_pos, peak_indexes):
    new_assignment_fft = dict()
    for pos in original_pos:
        new_pos, peak_index = find_closest_peak_fft(pos, support_pos, peak_indexes)
        new_assignment_fft[pos] = new_pos
    return new_assignment_fft


def construct_new_assignment_from_clusters(clusters_dict, threshold):
    new_assignment_fft = dict()
    for cluster in clusters_dict:
        if cluster['density'] > threshold:
            for member in cluster['members']:
                new_assignment_fft[member] = cluster['center']
    return new_assignment_fft


def find_closest_support_index(pos, support):
    min_diff = sys.maxint
    closest_sup = 0.0
    for sup in support:
        if abs(pos - sup) < min_diff:
            min_diff = abs(pos - sup)
            closest_sup = sup
    index = np.where(support == closest_sup)
    return index[0][0]


def construct_clusters(collection_per_chrom, support_per_chrom, density_per_chrom, peak_indexes):
    # list of {'density':cluster_value, 'center': cluster_center, 'members': [collection of points belonging to it]}
    clusters = dict()
    # set of cluster centers
    for i in range(len(collection_per_chrom)):
        for pos in collection_per_chrom[i]:
            cluster_center, peak_index = find_closest_peak_fft(pos, support_per_chrom[i], peak_indexes[i])
            # special case
            density = -1.0
            if peak_index == -1:
                closest_sup_ind = find_closest_support_index(pos, support_per_chrom[i])
                density = density_per_chrom[i][closest_sup_ind]
            else:
                density = density_per_chrom[i][peak_index]
            if density not in clusters:
                clusters[density] = {'density': density, 'center': cluster_center,
                                'members': [pos], 'chromosome': i}
            else:
                clusters[density]['members'].append(pos)
    # sort by density
    clusters_list = list()
    for key in clusters:
        clusters_list.append(clusters[key])
    clusters_list = sorted(clusters_list, key = lambda k: k['density'], reverse=True)
    return clusters_list


def external_edge(pos, orig_edges):
    # find an edge with this position
    for edge in orig_edges:
        if float(edge[0][1]) == pos and edge[0][0] != edge[1][0]:
            return True
        if float(edge[1][1]) == pos and edge[1][0] != edge[0][0]:
            return True
    return False


def internal_edge(pos1, pos2, orig_edges):
    for edge in orig_edges:
        if float(edge[0][1]) == pos1 and float(edge[1][1]) == pos2:
            return True
        if float(edge[1][1]) == pos1 and float(edge[0][1]) == pos2:
            return True
    return False


def test_new_assignment_with_correction(assignment, positions_by_chrom, orig_edges, sample, bandwidth, threshold_counter,
                                        path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/'):
    #fir the path
    path += 'b' + str(int(bandwidth)) + '/' + sample + '/t' + str(threshold_counter) + '/' + sample + '_metrics.csv'
    # reassign from values to keys
    min_dist_collection = list()
    max_dist_collection = list()
    for chrom in positions_by_chrom.keys():
        min_dist_non_joined = sys.maxint
        max_dist_joined = -sys.maxint - 1
        for pos1 in positions_by_chrom[chrom]:
            for pos2 in positions_by_chrom[chrom]:
                if pos1 == pos2 or pos1 not in assignment or pos2 not in assignment:
                    continue
                # if two positions assigned to the same destination - find out if the distance is bigger than the max
                if assignment[pos1] == assignment[pos2] \
                        and external_edge(pos1, orig_edges) \
                        and external_edge(pos2, orig_edges)\
                        and abs(pos1 - pos2) > max_dist_joined:
                    max_dist_joined = abs(pos1 - pos2)
                else:
                    # if two positions assigned to different destinations - find out if the distance is smaller than the min
                    if assignment[pos1] != assignment[pos2] and internal_edge(pos1, pos2, orig_edges)\
                            and abs(pos1 - pos2) < min_dist_non_joined:
                        min_dist_non_joined = abs(pos1 - pos2)

        if min_dist_non_joined < sys.maxint:
            min_dist_collection.append(min_dist_non_joined)
        if max_dist_joined > -sys.maxint - 1:
            max_dist_collection.append(max_dist_joined)
    if len(max_dist_collection) > 0:
        avg_max_joined = float(sum(max_dist_collection)) / len(max_dist_collection)
        max_max_joined = max(max_dist_collection)
    else:
        avg_max_joined = None
        max_max_joined = None
    if len(min_dist_collection) > 0:
        avg_min_non_joined = float(sum(min_dist_collection)) / len(min_dist_collection)
        min_min_non_joined = min(min_dist_collection)
    else:
        avg_min_non_joined = None
        min_min_non_joined = None
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'w+') as csvfile:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'metric': 'Inter avg. max. dist.', 'value': avg_max_joined})
        writer.writerow({'metric': 'Inter max. max. dist.', 'value': max_max_joined})
        writer.writerow({'metric': 'Intra avg. min. dist.', 'value': avg_min_non_joined})
        writer.writerow({'metric': 'Intra min. min. dist.', 'value': min_min_non_joined})
    csvfile.close()
    metrics_dict_list = []
    metrics_dict_list.append({'metric': 'Inter avg. max. dist.', 'value': avg_max_joined})
    metrics_dict_list.append({'metric': 'Inter max. max. dist.', 'value': max_max_joined})
    metrics_dict_list.append({'metric': 'Intra avg. min. dist.', 'value': avg_min_non_joined})
    metrics_dict_list.append({'metric': 'Intra min. min. dist.', 'value': min_min_non_joined})
    # print 'Inter Average Max distance between joined: ' + str(avg_max_joined)
    # print 'Inter Max Max distance between joined: ' + str(max_max_joined) + ', chrom: ' + str(max_dist_collection.index(max_max_joined))
    # print 'Intra Average Min distance between non-joined: ' + str(avg_min_non_joined)
    # print 'Intra Min Min distance between non-joined: ' + str(min_min_non_joined) + ', chrom: ' + str(min_dist_collection.index(min_min_non_joined))
    return metrics_dict_list


def write_data_to_hbase(new_edges, metrics, bandwidth, threshold_counter, threshold, sample):
    # write new assignment
    connection = happybase.Connection()
    sample_data_table = connection.table('sample_data')
    # create row_key
    row_key = 'b' + str(int(bandwidth)) + 't' + str(threshold_counter) + 's' + sample
    # collect values
    values = {b's:name': sample.encode('utf-8'), b's:cancer': SAMPLE_CANCER[sample].encode('utf-8'),
              b'b:value': str(int(bandwidth)).encode('utf-8'),
              b't:counter': str(threshold_counter).encode('utf-8'), b't:value': str(threshold).encode('utf-8')}
    for metric in metrics:
        values[('m:' + metric['metric']).encode('utf-8')] = str(metric['value']).encode('utf-8')
    chr1 = []
    pos1 = []
    chr2 = []
    pos2 = []
    for edge in new_edges:
        chrom1, position1 = find_relative_position(edge[0], True)
        chr1.append(chrom1)
        pos1.append(position1)
        chrom2, position2 = find_relative_position(edge[1], True)
        chr2.append(chrom2)
        pos2.append(position2)
    values[b'd:chr1'] = str(chr1).encode('utf-8')
    values[b'd:pos1'] = str(pos1).encode('utf-8')
    values[b'd:chr2'] = str(chr2).encode('utf-8')
    values[b'd:pos2'] = str(pos2).encode('utf-8')
    print  len(chr1), len(chr2), len(pos1), len(pos2)
    # print values
    sample_data_table.put(row_key.encode('utf-8'), values)


mode_separate = True
chrom = 8

client = Config().get_client('dev')
# read the positions from the largest sample


def calc_clustering_coeff_simple(new_edges, numedges_no_loops):
    nx_g = nx.Graph()
    nx_g.add_edges_from(new_edges)
    print len(list(nx_g.edges()))
    E = float(numedges_no_loops)
    V = float(len(list(nx_g.nodes())))
    density = (2 * E)/( V * (V - 1) )
    print 'density', density
    print 'coeff', density * len(list(nx_g.edges()))
    coeff_total = 0.0
    for vertex in list(nx_g.nodes()):
        neighbors = nx_g[vertex].keys()
        coeff = 0.0
        edges_set = set()
        for neighbor in neighbors:
            if neighbor != vertex:
                for other_neighbor in neighbors:
                    if other_neighbor != neighbor:
                        if nx_g.has_edge(neighbor, other_neighbor)\
                                and (neighbor, other_neighbor) not in edges_set \
                                and (other_neighbor, neighbor) not in edges_set:
                            coeff += 1
                            edges_set.add((neighbor, other_neighbor))
        if len(neighbors) > 1:
            coeff = coeff * 2 / len(neighbors) / (len(neighbors) - 1)
        #print coeff
        coeff_total += coeff
    coeff_total /= len(list(nx_g.nodes()))
    print 'coeff simple', coeff_total
    return coeff_total, density


def calc_clustering_coeff_loops(new_edges):
    nx_g = nx.Graph()
    nx_g.add_edges_from(new_edges)
    #print len(list(nx_g.edges()))
    coeff_total = 0.0
    for vertex in list(nx_g.nodes()):
        neighbors = nx_g[vertex].keys()
        coeff = 0.0
        edges_set = set()
        for neighbor in neighbors:
            for other_neighbor in neighbors:
                if nx_g.has_edge(neighbor, other_neighbor)\
                        and (neighbor, other_neighbor) not in edges_set \
                        and (other_neighbor, neighbor) not in edges_set:
                    coeff += 1
                    edges_set.add((neighbor, other_neighbor))
        if len(neighbors) > 0:
            coeff /= 2 * len(neighbors) * (len(neighbors) - 1) + len(neighbors)
        #print coeff
        coeff_total += coeff
    coeff_total /= len(list(nx_g.nodes()))
    print 'coeff with loops', coeff_total
    return coeff_total


def calc_max_pattern_size(new_edges):
    g = nx.Graph()
    g.add_edges_from(new_edges)
    conn_comp = nx.connected_component_subgraphs(g)
    max_comp = max(conn_comp, key=lambda graph: len(list(graph.edges())))
    print 'MAX CONN COMP', len(list(max_comp.edges()))


for bandwidth in [50000, 150000]:#BANDWIDTH_CANDIDATES:
    for sample in ['ec4d4cbc-d5d1-418d-a292-cad9576624fd']:#, '0448206f-3ade-4087-b1a9-4fb2d14e1367']:
        #['1ac15380-04a2-42dd-8ade-28556a570e80', '931b24da-5d6d-4c2d-8de9-ef32d6eb8565', '4c59fb2d-21b6-4b09-8174-6102de736e4d', '45e16b70-c3ec-493e-86d1-505ffdf5056c', 'bc0dee07-de20-44d6-be65-05af7e63ac96', 'a85cf239-ff51-46e7-9b88-4c2cb49c66b9', 'f83fc777-5416-c3e9-e040-11ac0d482c8e','a92023de-5c97-4bf2-aa3c-0e768d7c5ece', 'b27d75ba-5989-4200-bfe9-f1b7d7cf8008', '35c797fd-ca81-4cef-b6c4-7e3776f661b3',
                  # '9880c3c9-5685-42a7-8fe9-7585ea1a1d37', 'a67f4531-99ef-43df-82f5-f6abc4b11826', 'ea1cac20-88c1-4257-9cdb-d2890eb2e123', '9c681cd9-25fb-42ac-aa6b-bb962882fa22', '8ea666b7-2b6e-4df8-9a9d-b8265b9749b4', 'cdbbd701-9c05-4f9e-923d-06039dd8a04d', 'b38d0777-4901-48b8-9cdc-33b7f13a424f','ec4d4cbc-d5d1-418d-a292-cad9576624fd', '0448206f-3ade-4087-b1a9-4fb2d14e1367']:
            #['9880c3c9-5685-42a7-8fe9-7585ea1a1d37', 'a67f4531-99ef-43df-82f5-f6abc4b11826', 'ea1cac20-88c1-4257-9cdb-d2890eb2e123', '9c681cd9-25fb-42ac-aa6b-bb962882fa22', '8ea666b7-2b6e-4df8-9a9d-b8265b9749b4', 'cdbbd701-9c05-4f9e-923d-06039dd8a04d', 'b38d0777-4901-48b8-9cdc-33b7f13a424f','ec4d4cbc-d5d1-418d-a292-cad9576624fd', '0448206f-3ade-4087-b1a9-4fb2d14e1367']:# ['ec4d4cbc-d5d1-418d-a292-cad9576624fd','0448206f-3ade-4087-b1a9-4fb2d14e1367']:#SAMPLES:
        print 'BANDWIDTH: ' + str(int(bandwidth)) + ', SAMPLE: ' + sample
        edges = load_edges_2d(path= '/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/' + sample + '_positions.csv')
        ds_collection = convert_to_2d_array(edges)
        chromosome_list = list(chromosome for chromosome in ds_collection.keys())

        x_plot_y = []
        x_plot_y_sep = []
        for i in range(0, 24):
            if i in ds_collection:
                ds = ds_collection[i]
                x_plot_y_sep_tmp = []
                for x in ds:
                    y = func(x, 0)
                    x_plot_y.append(y)
                    x_plot_y_sep_tmp.append(y)
                x_plot_y_sep.append(x_plot_y_sep_tmp)

        # stastmodels.api with FFT
        # print 'Stats FFT'
        X_collection = []
        X_collection_sep = []
        X_collection_sup = []
        X_collection_sup_sep = []
        dens_collection = []
        dens_collection_sep = []
        indexes_collection = []
        indexes_collection_sep = []
        index_offset = 0
        for i in range(0, 24):
            if i in ds_collection:
                ds = ds_collection[i]
                X_collection.extend(ds)
                X_collection_sep.append(ds)
                dens_stats_fft = sm.nonparametric.KDEUnivariate(ds)
                dens_stats_fft.fit(bw=bandwidth, fft=True)
                X_collection_sup.extend(dens_stats_fft.support)
                X_collection_sup_sep.append(dens_stats_fft.support)
                dens_collection.extend(dens_stats_fft.density)
                dens_collection_sep.append(dens_stats_fft.density)
                indexes_stats_fft = peakutils.indexes(dens_stats_fft.density, thres=0.0, min_dist=0)
                # indexes_stats_fft = scipy.signal.find_peaks_cwt(dens_stats_fft.density, np.arange(0.001, 2))
                indexes_collection_sep.append(indexes_stats_fft)
                #print 'Chromosome ' + str(i) + ': ' + str(len(indexes_stats_fft)) + ' out of ' + str(len(ds)) + ' positions'
                for index in indexes_stats_fft:
                    indexes_collection.append(index_offset + index)
                index_offset += len(ds)
                first_chr = False
        plt.figure(num=None, figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
        if mode_separate:
            plt.plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
            plt.plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
            # plt.title('Sample ' + sample + '. Bandwidth = ' + str(bandwidth) + '. Stats FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])))
        else:
            plt.plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
            plt.plot(X_collection, x_plot_y, '+k')
            # plt.title('Sample ' + sample + '. Bandwidth = ' + str(bandwidth) + '. Stats FFT. NmbPeaks = ' + str(len(indexes_collection)))

        # HERE CONSTRUCT NEW CLUSTERS
        cluster_assignment = construct_clusters(X_collection_sep, X_collection_sup_sep, dens_collection_sep, indexes_collection_sep)
        #print len(cluster_assignment)#, cluster_assignment
        min_density = min(cluster_assignment, key=lambda x: x['density'])['density']
        max_density = max(cluster_assignment, key=lambda x: x['density'])['density']
        # # print str(min_density * 1000000000), str(max_density * 1000000000)
        # # create thresholds 30%, starting from 0
        thresholds = []
        third_of_clusters_nmb = int(round(float(len(cluster_assignment)) / 3.0))
        third_of_clusters = cluster_assignment[0:third_of_clusters_nmb]
        thresholds.append(third_of_clusters[len(third_of_clusters) - 1]['density'])
        third_of_clusters = cluster_assignment[third_of_clusters_nmb:2*third_of_clusters_nmb]
        thresholds.append(third_of_clusters[len(third_of_clusters) - 1]['density'])
        thresholds.append(0.0)

        # draw cluster assignment
        # n_groups = len(cluster_assignment)
        # # create plot
        # # fig, ax = plt.subplots()
        # index = np.arange(n_groups)
        # bar_width = 0.5
        # opacity = 0.8
        # # rects1 = plt.bar(index, [item['density'] for item in cluster_assignment], bar_width,
        # #                  alpha=opacity,
        # #                  color='b',
        # #                  label='Dens')
        #
        # # rects2 = plt.bar(index + bar_width, [len(item['members']) for item in cluster_assignment], bar_width,
        # #                  alpha=opacity,
        # #                  color='g',
        # #                  label='Size')
        # plt.xlabel('Clusters')
        # plt.ylabel('Density')
        # plt.title('B: ' + str(bandwidth) + ', Sample: ' + sample + ', Points: ' + str(len(x_plot_y)) + ', Clusters: ' + str(len(cluster_assignment)))
        # # plt.xticks(index + bar_width, range(len(cluster_assignment)))
        # plt.legend()
        # plt.tight_layout()
        # # plt.show()

        counter = 0
        for threshold in thresholds:
            if threshold == 0.0:
                print 'Threshold', threshold
                new_assignment = construct_new_assignment_from_clusters(cluster_assignment, threshold)
                new_edges = []
                for edge in edges:
                    if float(edge[0][1]) in new_assignment and float(edge[1][1]) in new_assignment:
                        new_edges.append((new_assignment[float(edge[0][1])], new_assignment[float(edge[1][1])]))
            # construct networkx graph

                num_edges = len(set(new_edges))
                print 'num_edges', num_edges
                num_loops = len([edge for edge in set(new_edges) if edge[0] == edge[1]])
                print 'num_loops', num_loops
                calc_clustering_coeff_simple(new_edges, num_edges - num_loops)
                calc_clustering_coeff_loops(new_edges)
                calc_max_pattern_size(new_edges)
            #  metrics = test_new_assignment_with_correction(new_assignment, ds_collection, edges, sample, bandwidth, counter)
        #     # print new_edges
        #     write_data_to_hbase(new_edges, metrics, bandwidth, counter, threshold, sample)
        #     write_undirect_input_file(new_edges,
        #                               path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
        #                                    + sample + '/t' + str(counter) + '/' + sample + '_new_assignment.csv')
        #     write_circos_input_file(new_edges,
        #                             path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
        #                                  + sample + '/t' + str(counter) + '/' + sample + '_for_circos.csv')
        #     write_init_files(client, sample, '/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
        #                                    + sample + '/t' + str(counter) + '/', bandwidth, counter)
        #     counter += 1
        #     # write_circos_input_file_orig(edges,
            #                              path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
            #                                   + sample + '/' + sample + '_orig_for_circos.csv')
        plt.show()
