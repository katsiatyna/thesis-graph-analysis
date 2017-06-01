# file for matching sample names to cancer types (all taken from NEW catalog from Luisa)
import os
from os import walk


dir_path = '/home/kkrasnas/Documents/thesis/pattern_mining/NEW'
fnames = []
sample_cancer = dict()
for (dirpath, dirnames, filenames) in walk(dir_path):
    fnames.extend(filenames)
    for fname in fnames:
        parts = fname.split('.')
        sample = parts[0]
        cancer = parts[1]
        sample_cancer[sample] = cancer
print sample_cancer
