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


def func(x, return_val):
    return return_val


CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]
mode_separate = True
chrom = 1
bandwidth = 2000.0
# read the positions from the largest sample
sample = '7d734d06-f2b1-4924-a201-620ac8084c49'
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
plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
if mode_separate:
    plt.plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=indexes_collection_sep[chrom])
    # local_max = scipy.signal.find_peaks_cwt(dens_collection_sep[chrom], np.arange(1, 2))
    # line2d = ax[3].plot(X_collection_sup_sep[chrom], dens_collection_sep[chrom], '-h', markevery=local_max)
    plt.plot(X_collection_sep[chrom], x_plot_y_sep[chrom], '+k')
    plt.title('Stats FFT. NmbPeaks = ' + str(len(indexes_collection_sep[chrom])))
else:
    plt.plot(X_collection_sup, dens_collection, '-h', markevery=indexes_collection)
    plt.plot(X_collection, x_plot_y, '+k')
    plt.title('Stats FFT. NmbPeaks = ' + str(len(indexes_collection)))

plt.show()
