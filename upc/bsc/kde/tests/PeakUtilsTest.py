import numpy
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot

centers = (30.5, 72.3)
x = numpy.linspace(0, 120, 121)
print x
y = (peakutils.gaussian(x, 5, centers[0], 3) +
    peakutils.gaussian(x, 7, centers[1], 10) +
    numpy.random.rand(x.size))
print y
pyplot.figure(figsize=(10,6))
pyplot.plot(x, y)
pyplot.title("Data with noise")

indexes = peakutils.indexes(y, thres=0.5, min_dist=30)
print(indexes)
print(x[indexes], y[indexes])
pyplot.figure(figsize=(10,6))
pplot(x, y, indexes)
pyplot.title('First estimate')
