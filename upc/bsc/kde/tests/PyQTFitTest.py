import numpy as np
#from scipy.stats import norm
from matplotlib import pylab as plt
from matplotlib import pyplot
from pyqt_fit import kde
#from detect_peaks import detect_peaks
import peakutils
from peakutils.plot import plot as pplot
import csv
import sys


#f = norm(loc=0, scale=1)
#x = f.rvs(500)
#xs = np.r_[-3:3:1024j]
#ys = f.pdf(xs)
#h = plt.hist(x, bins=30, normed=True, color=(0,.5,0,1), label='Histogram')
#plt.plot(xs, ys, 'r--', linewidth=2, label='$\mathcal{N}(0,1)$')
#plt.xlim(-3,3)
#plt.xlabel('X')
#plt.interactive(False)


def find_absolute_position(position, chromosome):
    if chromosome == 'X':
        index = 22
    else:
        if chromosome == 'Y':
            index = 23
        else:
            index = int(chromosome) - 1
    return long(CHR_MAP[index]) + long(position)


def load_edges(path='/home/kkrasnas/Documents/thesis/pattern_mining/PositionsTest.csv'):
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['Pos_BKP_1', 'Chr_BKP_1', 'Pos_BKP_2', 'Chr_BKP_2'])
        next(csvfile)
        positions = []
        for row in reader:
            positions.append((find_absolute_position(row['Pos_BKP_1'], row['Chr_BKP_1']), find_absolute_position(row['Pos_BKP_2'], row['Chr_BKP_2'])))
        return positions

def convert_to_flat_array(edges):
    positions = []
    for edge in edges:
        positions.append(edge[0])
        positions.append(edge[1])
    ds = sorted(set(map(float,positions)))
    return ds

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


def construct_new_assignment(original_pos, peak_indexes, edges):
    new_assignment = []
    for edge in edges:
        new_assignment.append((original_pos[find_closest_peak(original_pos.index(float(edge[0])), peak_indexes)], original_pos[find_closest_peak(original_pos.index(float(edge[1])), peak_indexes)]))
    return new_assignment


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

#info chromosome
CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]

cxy = 0
startxy = None
endxy = None
for  i in range(len(CHR_MAP)):
    startxy =(startxy,cxy+1)
    cxy = cxy + CHR_MAP[i]
    endxy = (endxy,cxy)

#labelxy = paste("chr",c(sprintf("%02d",seq(1:22)),"X","Y"),sep='')


# read the positions from the largest sample
edges = load_edges()
ds = convert_to_flat_array(edges)

# calculate density estimation
est = kde.KDE1D(ds)
est.bandwidth =10000 #10k window?
estimation = est(ds)
#est.lower = 25319510
#est.upper = 120155230

# find peaks in the density
indexes = peakutils.indexes(estimation, thres=0.01, min_dist=0)
#plt.plot(ds, est(ds), label='Estimate (bw={:.3g})'.format(est.bandwidth))
pyplot.figure(figsize=(20,5))
pyplot.plot(ds, estimation, '-bh', markevery=indexes)
pyplot.legend(loc='best')
pyplot.title('Density Peaks. NmbPeaks = ' + str(len(indexes)))
pyplot.show()

# assign each point to closest peak and rewrite the edges



new_assignment = construct_new_assignment(ds, indexes, edges)

# write new assignment
with open('/home/kkrasnas/Documents/thesis/pattern_mining/new_assignment.csv', 'wb') as csvfile:
    fieldnames = ['pos_1', 'pos_2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in new_assignment:
        writer.writerow({'pos_1': row[0], 'pos_2': row[1]})


 #use peaksutils or
# f = norm(loc=0, scale=1)
# xs = np.r_[-3:3:1024j]
# nbins = 20
# x = f.rvs(1000*1000).reshape(1000,1000)
# hs = np.empty((1000, nbins), dtype=float)
# kdes = np.empty((1000, 1024), dtype=float)
# hs[0], edges = np.histogram(x[0], bins=nbins, range=(-3,3), density=True)
# mod = kde.KDE1D(x[0])
# mod.fit()  # Force estimation of parameters
# mod.bandwidth = mod.bandwidth  # Prevent future recalculation
# kdes[0] = mod(xs)
# for i in range(1, 1000):
#    hs[i] = np.histogram(x[i], bins=nbins, range=(-3,3), density=True)[0]
#    mod.xdata = x[i]
#    kdes[i] = mod(xs)
# h_mean = hs.mean(axis=0)
# h_ci = np.array(np.percentile(hs, (5, 95), axis=0))
# h_err = np.empty(h_ci.shape, dtype=float)
# h_err[0] = h_mean - h_ci[0]
# h_err[1] = h_ci[1] - h_mean
# kde_mean = kdes.mean(axis=0)
# kde_ci = np.array(np.percentile(kdes, (5, 95), axis=0))
# width = edges[1:]-edges[:-1]
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# ax1.bar(edges[:-1], h_mean, yerr=h_err, width = width, label='Histogram',
#         facecolor='g', edgecolor='k', ecolor='b')
# ax1.plot(xs, f.pdf(xs), 'r--', lw=2, label='$\mathcal{N}(0,1)$')
# ax1.set_xlabel('X')
# ax1.set_xlim(-3,3)
# ax1.legend(loc='best')
# ax2 = fig.add_subplot(1,2,2)
# ax2.fill_between(xs, kde_ci[0], kde_ci[1], color=(0,1,0,.5), edgecolor=(0,.4,0,1))
# ax2.plot(xs, kde_mean, 'k', label='KDE (bw = {:.3g})'.format(mod.bandwidth))
# ax2.plot(xs, f.pdf(xs), 'r--', lw=2, label='$\mathcal{N}(0,1)$')
# ax2.set_xlabel('X')
# ax2.legend(loc='best')
# ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
# ax2.set_ylim(0, ymax)
# ax1.set_ylim(0, ymax)
# ax1.set_title('Histogram, max variation = {:.3g}'.format((h_ci[1] - h_ci[0]).max()))
# ax2.set_title('KDE, max variation = {:.3g}'.format((kde_ci[1] - kde_ci[0]).max()))
# fig.suptitle('Comparison Histogram vs. KDE')
# plt.show()

