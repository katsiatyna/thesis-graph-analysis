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


def diff(pos, peak):
    return abs(pos - peak)


def find_closest_peak(i, peak_indexes):
    min_diff = sys.maxint
    closest_peak = -1
    for peak in peak_indexes:
        if(diff(i, peak) < min_diff):
            min_diff = diff(i,peak)
            closest_peak = peak
    return closest_peak


def find_closest_peak_fft(pos, support_pos, peak_indexes):
    min_diff = sys.maxint
    new_pos = pos
    for peak in peak_indexes:
        if(diff(pos, support_pos[peak]) < min_diff):
            min_diff = diff(pos, support_pos[peak])
            new_pos = support_pos[peak]
    return new_pos


def construct_new_assignment(original_pos, peak_indexes, edges):
    new_assignment = []
    for edge in edges:
        new_assignment.append((original_pos[find_closest_peak(original_pos.index(float(edge[0])), peak_indexes)],
                               original_pos[find_closest_peak(original_pos.index(float(edge[1])), peak_indexes)]))
    return new_assignment


def construct_new_assignment_fft(original_pos, support_pos, peak_indexes):
    new_assignment_fft = dict()
    for pos in original_pos:
        new_pos = find_closest_peak_fft(pos, support_pos, peak_indexes)
        new_assignment_fft[pos] = new_pos
    return new_assignment_fft


def func(x, return_val):
    return return_val


def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale


def write_xgraph_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/graphXdir.txt'):
    f = open(path, 'w')
    for row in assignment:
        f.write(str(long(row[0])) + ' ' + str(long(row[1])) + '\n')
        # write backward edges for GraphX
        if row[0] != row[1]:
            f.write(str(long(row[1])) + ' ' + str(long(row[0])) + '\n')
    f.close()


def write_undirect_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv'):
    # write new assignment
    with open(path, 'wb') as csvfile:
        fieldnames = ['pos_1', 'pos_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in assignment:
            writer.writerow({'pos_1': row[0], 'pos_2': row[1]})


def write_circos_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_for_circos.csv'):
    # write new assignment
    with open(path, 'wb') as csvfile:
        fieldnames = ['source_id','source_breakpoint','target_id','target_breakpoint','source_label','target_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        print 'New assignment before deduplication: ' + str(len(assignment))
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
        print 'New assignment after deduplication: ' + str(len(assignment_deduplicated))
        writer.writeheader()
        for row in assignment_deduplicated:
            source_chr, source_rel_pos = find_relative_position(row[0])
            target_chr, target_rel_pos = find_relative_position(row[1])
            writer.writerow({'source_id': source_chr, 'source_breakpoint': source_rel_pos,
                             'target_id':target_chr, 'target_breakpoint': target_rel_pos,
                             'source_label':'', 'target_label':''})


def write_circos_input_file_orig(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_orig_for_circos.csv'):
    # write new assignment
    with open(path, 'wb') as csvfile:
        fieldnames = ['source_id','source_breakpoint','target_id','target_breakpoint','source_label','target_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in assignment:
            source_chr, source_rel_pos = find_relative_position(row[0][1])
            target_chr, target_rel_pos = find_relative_position(row[1][1])
            writer.writerow({'source_id': source_chr, 'source_breakpoint': source_rel_pos,
                             'target_id':target_chr, 'target_breakpoint': target_rel_pos,
                             'source_label':'', 'target_label':''})


def write_lg_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/graph.lg'):
    f = open(path, 'w')
    f.write('# t 1\n')
    vertices = set()
    for row in assignment:
        vertices.add(row[0])
        vertices.add(row[1])
    vertices = sorted(list(vertices))
    assignment_sorted = sorted(assignment,key=lambda x: x[0])
    for i in range(len(vertices)):
        f.write('v ' + str(i) + ' 0\n')

    for row in assignment_sorted:
        f.write('e ' + str(vertices.index(row[0])) + ' ' + str(vertices.index(row[1])) + ' 1\n')

    f.close()


def test_new_assignment(assignment, positions_by_chrom):
    # reassign from values to keys
    min_dist_collection = list()
    max_dist_collection = list()
    for chrom in positions_by_chrom.keys():
        min_dist_non_joined = sys.maxint
        max_dist_joined = -sys.maxint - 1
        for pos1 in positions_by_chrom[chrom]:
            for pos2 in positions_by_chrom[chrom]:
                if pos1 == pos2:
                    continue
                # if two positions assigned to the same destination - find out if the distance is bigger than the max
                if assignment[pos1] == assignment[pos2] and abs(pos1 - pos2) > max_dist_joined:
                    max_dist_joined = abs(pos1 - pos2)
                else:
                    # if two positions assigned to different destinations - find out if the distance is smaller than the min
                    if assignment[pos1] != assignment[pos2] and abs(pos1 - pos2) < min_dist_non_joined:
                        min_dist_non_joined = abs(pos1 - pos2)

        if min_dist_non_joined < sys.maxint:
            min_dist_collection.append(min_dist_non_joined)
        if max_dist_joined > -sys.maxint - 1:
            max_dist_collection.append(max_dist_joined)
    avg_max_joined = float(sum(max_dist_collection)) / len(max_dist_collection)
    max_max_joined = max(max_dist_collection)
    avg_min_non_joined = float(sum(min_dist_collection)) / len(min_dist_collection)
    min_min_non_joined = min(min_dist_collection)
    print 'Average Max distance between joined: ' + str(avg_max_joined)
    print 'Max Max distance between joined: ' + str(max_max_joined) + ', chrom: ' + str(max_dist_collection.index(max_max_joined))
    print 'Average Min distance between non-joined: ' + str(avg_min_non_joined)
    print 'Min Min distance between non-joined: ' + str(min_min_non_joined) + ', chrom: ' + str(min_dist_collection.index(min_min_non_joined))


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


def test_new_assignment_with_correction(assignment, positions_by_chrom, orig_edges, sample,
                                        path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/'):
    #fir the path
    path += sample + '/' + sample + '_metrics.csv'
    # reassign from values to keys
    min_dist_collection = list()
    max_dist_collection = list()
    for chrom in positions_by_chrom.keys():
        min_dist_non_joined = sys.maxint
        max_dist_joined = -sys.maxint - 1
        for pos1 in positions_by_chrom[chrom]:
            for pos2 in positions_by_chrom[chrom]:
                if pos1 == pos2:
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
    avg_max_joined = float(sum(max_dist_collection)) / len(max_dist_collection)
    max_max_joined = max(max_dist_collection)
    avg_min_non_joined = float(sum(min_dist_collection)) / len(min_dist_collection)
    min_min_non_joined = min(min_dist_collection)
    with open(path, 'wb') as csvfile:
        fieldnames = ['metric','value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'metric': 'Inter avg. max. dist.', 'value': avg_max_joined})
        writer.writerow({'metric': 'Inter max. max. dist.', 'value': max_max_joined})
        writer.writerow({'metric': 'Intra avg. min. dist.', 'value': avg_min_non_joined})
        writer.writerow({'metric': 'Intra min. min. dist.', 'value': min_min_non_joined})
    csvfile.close()
    print 'Inter Average Max distance between joined: ' + str(avg_max_joined)
    print 'Inter Max Max distance between joined: ' + str(max_max_joined) + ', chrom: ' + str(max_dist_collection.index(max_max_joined))
    print 'Intra Average Min distance between non-joined: ' + str(avg_min_non_joined)
    print 'Intra Min Min distance between non-joined: ' + str(min_min_non_joined) + ', chrom: ' + str(min_dist_collection.index(min_min_non_joined))


def find_peaks_indexes(density_y):
    peak_indexes = list()
    for i in range(len(density_y)):
        if i == 0 and density_y[i] > density_y[i+1]:
            peak_indexes.append(i)
            continue
        if i == len(density_y) - 1 and density_y[i] > density_y[i-1]:
            peak_indexes.append(i)
            continue
        if density_y[i-1] < density_y[i] > density_y[i+1]:
            peak_indexes.append(i)
            continue
    return peak_indexes



# info chromosome
CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]
mode_separate = True
chrom = 1
bandwidth = 50000.0
# read the positions from the largest sample
sample = 'ea1cac20-88c1-4257-9cdb-d2890eb2e123'
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
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=False, squeeze=True,
             subplot_kw=None, gridspec_kw=None, figsize=(35, 15))

# scipy
print 'SciPy'
X_collection = []
X_collection_sep = []
log_dens_collection = []
log_dens_collection_sep = []
indexes_scipy_collection = []
indexes_scipy_collection_sep = []
index_offset = 0
margins = [0]
for i in range(0, 24):
    if i in ds_collection:
        ds = ds_collection[i]
        X = np.array(ds)
        X_collection.extend(X)
        X_collection_sep.append(X)
        X_res = X.reshape(-1, 1)
        scipy_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_res)
        log_dens = scipy_kde.score_samples(X_res)
        log_dens_collection.extend(log_dens)
        log_dens_collection_sep.append(log_dens)
        indexes_scipy = peakutils.indexes(np.exp(log_dens), thres=0.0, min_dist=0)
        print 'Chromosome ' + str(i) + ': ' + str(len(indexes_scipy)) + ' out of ' + str(len(ds)) + ' positions'
        indexes_scipy_collection_sep.append(indexes_scipy)
        for index_scipy in indexes_scipy:
            indexes_scipy_collection.append(index_offset + index_scipy)
        index_offset += len(X)
        margins.append(index_offset - 1)

if(mode_separate):
    ax[0].plot(X_collection_sep[chrom], np.exp(log_dens_collection_sep[chrom]),  '-h', markevery=indexes_scipy_collection_sep[chrom])
    ax[0].plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    # ax[0].plot(X_collection, x_plot_y, '+r', markevery=margins)
    ax[0].set_title('Bandwidth = ' + str(int(bandwidth / 1000)) + 'kbp'
                    + ', sample chromosome = ' + str(chromosome_list[chrom])
                    + ', NmbPoints in chromosome = ' + str(len(x_plot_y_sep[chrom]))
                    + ': SciPy, NmbPeaks = ' + str(len(indexes_scipy_collection_sep[chrom])))
    ax[0].axes.get_xaxis().set_ticks([])
else:
    ax[0].plot(X_collection, np.exp(log_dens_collection),  '-h', markevery=indexes_scipy_collection)
    ax[0].plot(X_collection, x_plot_y, '+k')
    ax[0].plot(X_collection, x_plot_y, '+r', markevery=margins)
    ax[0].annotate('SciPy. NmbPeaks = ' + str(len(indexes_scipy_collection)), xy=get_axis_limits(ax[0]))
    ax[0].set_title('Bandwidth = ' + str(int(bandwidth / 1000)) + 'kbp, NmbPoints = ' + str(len(x_plot_y))
                    + ': SciPy, NmbPeaks = ' + str(len(indexes_scipy_collection)))
    ax[0].axes.get_xaxis().set_ticks([])

# pyqt-fit
# calculate density estimation
print 'PyQT-Fit'
X_collection = []
X_collection_sep = []
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
        est = kde.KDE1D(ds)
        #est.kernel = kernels.normal_kernel1d()
        est.bandwidth =bandwidth #10k window?
        estimation = est(ds)
        dens_collection.extend(estimation)
        dens_collection_sep.append(estimation)
        #est.lower = 25319510
        #est.upper = 120155230
        indexes = peakutils.indexes(estimation, thres=0.0, min_dist=0)
        indexes_collection_sep.append(indexes)
        print 'Chromosome ' + str(i) + ': ' + str(len(indexes))+ ' out of ' + str(len(ds)) + ' positions'
        for index in indexes:
            indexes_collection.append(index_offset + index)
        index_offset += len(ds)
        first_chr = False
#plt.plot(ds, est(ds), label='Estimate (bw={:.3g})'.format(est.bandwidth))
if mode_separate:
    ax[1].plot(X_collection_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
    ax[1].plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    ax[1].set_title('PyQT-Fit. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])))
    ax[1].axes.get_xaxis().set_ticks([])
else:
    ax[1].plot(X_collection, dens_collection, '-h', markevery=indexes_collection)
    ax[1].plot(X_collection, x_plot_y, '+k')
    ax[1].set_title('PyQT-Fit. NmbPeaks = ' + str(len(indexes_collection)))
    ax[1].axes.get_xaxis().set_ticks([])

# stastmodels.api without FFT
print 'Stats no FFT'
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
        dens_stats = sm.nonparametric.KDEUnivariate(ds)
        dens_stats.fit(bw=bandwidth, fft=False)
        X_collection.extend(ds)
        X_collection_sep.append(ds)
        X_collection_sup.extend(dens_stats.support)
        X_collection_sup_sep.append(dens_stats.support)
        dens_collection.extend(dens_stats.density)
        dens_collection_sep.append(dens_stats.density)
        indexes_stats = peakutils.indexes(dens_stats.density, thres=0.0, min_dist=0)
        indexes_collection_sep.append(indexes_stats)
        print 'Chromosome ' + str(i) + ': ' + str(len(indexes_stats))+ ' out of ' + str(len(ds)) + ' positions'
        for index in indexes_stats:
            indexes_collection.append(index_offset + index)
        index_offset += len(ds)
        first_chr = False
if mode_separate:
    ax[2].plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
    ax[2].plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    ax[2].set_title('Stats No FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])))
    ax[2].axes.get_xaxis().set_ticks([])
else:
    ax[2].plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
    ax[2].plot(X_collection, x_plot_y, '+k')
    ax[2].set_title('Stats No FFT. NmbPeaks = ' + str(len(indexes_collection)))
    ax[2].axes.get_xaxis().set_ticks([])

# stastmodels.api with FFT
print 'Stats FFT'
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
        print 'Chromosome ' + str(i) + ': ' + str(len(indexes_stats_fft)) + ' out of ' + str(len(ds)) + ' positions'
        for index in indexes_stats_fft:
            indexes_collection.append(index_offset + index)
        index_offset += len(ds)
        first_chr = False
if mode_separate:
    line2d = ax[3].plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
    # local_max = scipy.signal.find_peaks_cwt(dens_collection_sep[chrom], np.arange(1, 2))
    # line2d = ax[3].plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=local_max)
    ax[3].plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    ax[3].set_title('Stats FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])))
else:
    ax[3].plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
    ax[3].plot(X_collection, x_plot_y, '+k')
    ax[3].set_title('Stats FFT. NmbPeaks = ' + str(len(indexes_collection)))


# # assign each point to closest peak and rewrite the edges
# new_assignment = construct_new_assignment(ds, indexes, edges)
new_assignment = dict()
for i in range(len(ds_collection)):
    new_assignment.update(construct_new_assignment_fft(X_collection_sep[i], X_collection_sup_sep[i], indexes_collection_sep[i]))
new_edges = []
for edge in edges:
    new_edges.append((new_assignment[float(edge[0][1])], new_assignment[float(edge[1][1])]))
# print new_edges
write_undirect_input_file(new_edges, path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/'
                                          + sample + '_new_assignment.csv')
write_circos_input_file(new_edges, path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/'
                                          + sample + '_for_circos.csv')
write_circos_input_file_orig(edges, path='/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/'
                                          + sample + '_orig_for_circos.csv')
# # write_xgraph_input_file(new_assignment)
# # write_lg_input_file(new_assignment)


test_new_assignment(new_assignment, ds_collection)
#test_new_assignment_with_correction(new_assignment, ds_collection, edges, sample)
pyplot.show()
