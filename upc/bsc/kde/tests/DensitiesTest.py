import numpy as np
from matplotlib import pylab as plt
from matplotlib import pyplot
from pyqt_fit import kde, kernels
import peakutils
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
import csv
import sys


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
        reader = csv.DictReader(csvfile, fieldnames=['Chr_BKP_1', 'Pos_BKP_1', 'Chr_BKP_2', 'Pos_BKP_2'])
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

bandwidth = 1000000.0
# read the positions from the largest sample
edges = load_edges()
ds = convert_to_flat_array(edges)


x_plot_y = []
for x in ds:
    x_plot_y.append(func(ds, 0))
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=False, squeeze=True,
             subplot_kw=None, gridspec_kw=None, figsize=(20, 15))
pyplot.title('Bandwidth = ' + str(int(bandwidth/1000)) + 'kbp, NmbPoints = ' + str(len(ds)))
# scipy
X = np.array(ds)
X_res = X.reshape(-1,1)
scipy_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_res)
log_dens = scipy_kde.score_samples(X_res)
indexes_scipy = peakutils.indexes(np.exp(log_dens), thres=0.0, min_dist=0)
ax[0].plot(X, np.exp(log_dens),  '-h', markevery=indexes_scipy)
ax[0].plot(ds, x_plot_y, '+k')
ax[0].annotate('SciPy. NmbPeaks = ' + str(len(indexes_scipy)), xy=get_axis_limits(ax[0]))


# pyqt-fit
# calculate density estimation
est = kde.KDE1D(ds)
#est.kernel = kernels.normal_kernel1d()
est.bandwidth =bandwidth #10k window?
estimation = est(ds)
#est.lower = 25319510
#est.upper = 120155230
indexes = peakutils.indexes(estimation, thres=0.0, min_dist=0)
#plt.plot(ds, est(ds), label='Estimate (bw={:.3g})'.format(est.bandwidth))
ax[1].plot(ds, estimation, '-h', markevery=indexes)
ax[1].plot(ds, x_plot_y, '+k')
ax[1].annotate('PyQT-Fit. NmbPeaks = ' + str(len(indexes)), xy=get_axis_limits(ax[1]))

# stastmodels.api without FFT
dens_stats = sm.nonparametric.KDEUnivariate(ds)
dens_stats.fit(bw=bandwidth, fft=False)
indexes_stats = peakutils.indexes(dens_stats.density, thres=0.1, min_dist=0)

ax[2].plot(dens_stats.support, dens_stats.density, '-h', markevery=indexes_stats)
ax[2].plot(ds, x_plot_y, '+k')
ax[2].annotate('Stats No FFT. NmbPeaks = ' + str(len(indexes_stats)), xy=get_axis_limits(ax[2]))

# stastmodels.api without FFT
dens_stats_fft = sm.nonparametric.KDEUnivariate(ds)
dens_stats_fft.fit(bw=bandwidth, fft=True)
indexes_stats_fft = peakutils.indexes(dens_stats_fft.density, thres=0.1, min_dist=0)

ax[3].plot(dens_stats_fft.support, dens_stats_fft.density, '-h', markevery=indexes_stats_fft)
ax[3].plot(ds, x_plot_y, '+k')
ax[3].annotate('Stats FFT. NmbPeaks = ' + str(len(indexes_stats_fft)), xy=get_axis_limits(ax[3]))




pyplot.show()


# assign each point to closest peak and rewrite the edges
new_assignment = construct_new_assignment(ds, indexes, edges)
write_undirect_input_file(new_assignment)
write_xgraph_input_file(new_assignment)
write_lg_input_file(new_assignment)
