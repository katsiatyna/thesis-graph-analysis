import numpy as np
from matplotlib import pylab as plt
from matplotlib import pyplot
from pyqt_fit import kde, kernels
import peakutils
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
import csv
import sys


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
        for i in range(index + 1):
            offset += long(CHR_MAP[i])
    return offset + long(position)


def load_edges_2d(path='/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/7d734d06-f2b1-4924-a201-620ac8084c49_positions.csv'):
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['Chr_BKP_1', 'Pos_BKP_1', 'Chr_BKP_2', 'Pos_BKP_2'])
        next(csvfile)
        positions = []
        for row in reader:
            positions.append(((chr_number(row['Chr_BKP_1']), find_absolute_position(row['Pos_BKP_1'], row['Chr_BKP_1'])), (chr_number(row['Chr_BKP_2']), find_absolute_position(row['Pos_BKP_2'], row['Chr_BKP_2']))))
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
    new_assignment = dict()
    for pos in original_pos:
        new_pos = find_closest_peak_fft(pos, support_pos, peak_indexes)
        new_assignment[pos] = new_pos
    return new_assignment


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


def write_undirect_input_file(assignment, path='/home/kkrasnas/Documents/thesis/pattern_mining/new_assignment.csv'):
    # write new assignment
    with open(path, 'wb') as csvfile:
        fieldnames = ['pos_1', 'pos_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in assignment:
            writer.writerow({'pos_1': row[0], 'pos_2': row[1]})


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

# ds = sorted(map(float, [80407479,50425934,82653054,132506654,132512574,8137459,8137565,55013506,21479214,41670179,43294258,45820099,
#                  45864568,45880888,47120476,104483915,41745869,32087843,85867303,86386946,7537761,59316711,71262976,55866224,
#                  185147302,234917206,234917491,24489763,24755242,30971243,39867712,39883066,11421883,11836267,29602419,115902151,
#                  116239951,119250739,121536351,121536416,112891331,170706563,47174222,111733407,4188644,4357547,36146852,
#                  57974035,61873557,62262194,66027930,83251943,141212309,25319512,26175897,26179530,26180930,32931461,41833641,
#                  100418583,120152091,71813442,71813782,58109172,80410698,50429962,82655509,53374859,134457664,124070384,
#                         124070269,4344156,16887282,41707033,10089846,41686214,68397217,12626390,42584288,104485559,41748503,
#                         28193653,86711648,87254181,8091000,59368935,41709592,16764011,185149442,42851895,42870301,24776966,28210105,
#                         30996135,42872048,42871868,193409197,194134696,86761278,144636867,118612945,151685509,144990021,
#                         144990143,112893006,170708906,47199123,111964714,4266657,4360229,36189429,57975730,61875756,62265611,
#                         91154657,83253194,141224270,32935734,32930323,32929935,32934432,32934337,42831908,100420068,120155220,
#                         89952391,89952299,58110742]))

# info chromosome
CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]
mode_separate = True
chrom = 0
bandwidth = 50000.0
# read the positions from the largest sample
edges = load_edges_2d()
ds_collection = convert_to_2d_array(edges)


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
pyplot.title('Bandwidth = ' + str(int(bandwidth/1000)) + 'kbp, NmbPoints = ' + str(len(x_plot_y)))
# scipy
print 'SciPy'
X_collection = []
X_collection_sep = []
log_dens_collection = []
log_dens_collection_sep = []
indexes_scipy_collection = []
indexes_scipy_collection_sep = []
index_offset = 0
first_chr = True
margins = [0]
for i in range(0, 24):
    if i in ds_collection:
        ds = ds_collection[i]
        X = np.array(ds)
        X_collection.extend(X)
        X_collection_sep.append(X)
        X_res = X.reshape(-1,1)
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
        first_chr = False

if(mode_separate):
    ax[0].plot(X_collection_sep[chrom], np.exp(log_dens_collection_sep[chrom]),  '-h', markevery=indexes_scipy_collection_sep[chrom])
    ax[0].plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    # ax[0].plot(X_collection, x_plot_y, '+r', markevery=margins)
    ax[0].annotate('SciPy. NmbPeaks = ' + str(len(indexes_scipy_collection_sep[chrom])), xy=get_axis_limits(ax[0]))
else:
    ax[0].plot(X_collection, np.exp(log_dens_collection),  '-h', markevery=indexes_scipy_collection)
    ax[0].plot(X_collection, x_plot_y, '+k')
    # ax[0].plot(X_collection, x_plot_y, '+r', markevery=margins)
    ax[0].annotate('SciPy. NmbPeaks = ' + str(len(indexes_scipy_collection)), xy=get_axis_limits(ax[0]))


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
first_chr = True
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
    ax[1].annotate('PyQT-Fit. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])), xy=get_axis_limits(ax[1]))
else:
    ax[1].plot(X_collection, dens_collection, '-h', markevery=indexes_collection)
    ax[1].plot(X_collection, x_plot_y, '+k')
    ax[1].annotate('PyQT-Fit. NmbPeaks = ' + str(len(indexes_collection)), xy=get_axis_limits(ax[1]))

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
first_chr = True
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
    ax[2].annotate('Stats No FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])), xy=get_axis_limits(ax[2]))
else:
    ax[2].plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
    ax[2].plot(X_collection, x_plot_y, '+k')
    ax[2].annotate('Stats No FFT. NmbPeaks = ' + str(len(indexes_collection)), xy=get_axis_limits(ax[2]))

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
first_chr = True
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
        indexes_collection_sep.append(indexes_stats_fft)
        print 'Chromosome ' + str(i) + ': ' + str(len(indexes_stats_fft))+ ' out of ' + str(len(ds)) + ' positions'
        for index in indexes_stats_fft:
            indexes_collection.append(index_offset + index)
        index_offset += len(ds)
        first_chr = False
if mode_separate:
    ax[3].plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
    ax[3].plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    ax[3].annotate('Stats FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])), xy=get_axis_limits(ax[3]))
else:
    ax[3].plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
    ax[3].plot(X_collection, x_plot_y, '+k')
    ax[3].annotate('Stats FFT. NmbPeaks = ' + str(len(indexes_collection)), xy=get_axis_limits(ax[3]))







# # assign each point to closest peak and rewrite the edges
# new_assignment = construct_new_assignment(ds, indexes, edges)
new_assignment = dict()
for i in range(len(ds_collection)):
    new_assignment.update(construct_new_assignment_fft(X_collection_sep[i], X_collection_sup_sep[i], indexes_collection_sep[i]))
new_edges = []
for edge in edges:
    new_edges.append((new_assignment[float(edge[0][1])], new_assignment[float(edge[1][1])]))
print new_edges
# write_undirect_input_file(new_assignment)
# # write_xgraph_input_file(new_assignment)
# # write_lg_input_file(new_assignment)

pyplot.show()
