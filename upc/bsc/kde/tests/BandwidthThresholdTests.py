import numpy as np
from matplotlib import pylab as plt
from matplotlib import pyplot
from pyqt_fit import kde, kernels
import peakutils
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
import csv
import sys
from scipy.signal import argrelextrema
import scipy
from pypeaks import Data, Intervals
from upc.bsc.Constants import SAMPLES, CHR_MAP


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


def find_relative_position(position):
    # index = chr_number(chromosome)
    offset = 0L
    for i in range(len(CHR_MAP) + 1):
        if offset < position < offset + long(CHR_MAP[i]):
            chrom_str = chr_str(i)
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
    # write new assignment
    with open(path, 'w+') as csvfile:
        fieldnames = ['pos_1', 'pos_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in assignment:
            writer.writerow({'pos_1': row[0], 'pos_2': row[1]})


def write_circos_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_for_circos.csv'):
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
    for peak in peak_indexes:
        if(diff(pos, support_pos[peak]) < min_diff):
            min_diff = diff(pos, support_pos[peak])
            new_pos = support_pos[peak]
    return new_pos


def construct_new_assignment_fft(original_pos, support_pos, peak_indexes):
    new_assignment_fft = dict()
    for pos in original_pos:
        new_pos = find_closest_peak_fft(pos, support_pos, peak_indexes)
        new_assignment_fft[pos] = new_pos
    return new_assignment_fft



mode_separate = True
chrom = 1
bandwidth_candidates = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 150000.0]
# read the positions from the largest sample
for bandwidth in bandwidth_candidates:
    for sample in SAMPLES:
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
                # print 'Chromosome ' + str(i) + ': ' + str(len(indexes_stats_fft)) + ' out of ' + str(len(ds)) + ' positions'
                for index in indexes_stats_fft:
                    indexes_collection.append(index_offset + index)
                index_offset += len(ds)
                first_chr = False
        plt.figure(num=None, figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
        if mode_separate:
            plt.plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
            plt.plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
            plt.title('Sample ' + sample + '. Bandwidth = ' + str(bandwidth) + '. Stats FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])))
        else:
            plt.plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
            plt.plot(X_collection, x_plot_y, '+k')
            plt.title('Sample ' + sample + '. Bandwidth = ' + str(bandwidth) + '. Stats FFT. NmbPeaks = ' + str(len(indexes_collection)))
        new_assignment = dict()
        for i in range(len(ds_collection)):
            new_assignment.update(construct_new_assignment_fft(X_collection_sep[i], X_collection_sup_sep[i],
                                                               indexes_collection_sep[i]))
        new_edges = []
        for edge in edges:
            new_edges.append((new_assignment[float(edge[0][1])], new_assignment[float(edge[1][1])]))
        # print new_edges
        write_undirect_input_file(new_edges,
                                  path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
                                       + sample + '/' + sample + '_new_assignment.csv')
        write_circos_input_file(new_edges,
                                path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
                                     + sample + '/' + sample + '_for_circos.csv')
        # write_circos_input_file_orig(edges,
        #                              path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/b' + str(int(bandwidth)) + '/'
        #                                   + sample + '/' + sample + '_orig_for_circos.csv')
        # plt.show()
