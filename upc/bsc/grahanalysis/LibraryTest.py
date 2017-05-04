"""library.py for textparser app

Author: James Hohman
Copyright James Hohman 2015
"""
from collections import Counter
from django.contrib import messages
import HTMLParser
import string
import csv
import HDFSBrowserTest as hadoop
from django.contrib.staticfiles.templatetags.staticfiles import static

# Global.
# replace punctuation instead of remove
tr = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
           'ea1cac20-88c1-4257-9cdb-d2890eb2e123']


def collect_graphs(request):
    json_dict = {}
    results_all = hadoop.return_analysis_results()
    for sample in results_all.keys():
        json_dict[sample] = {}
        for size in results_all[sample]:
            for pattern in results_all[sample][size].keys():
                json_dict[sample][pattern] = {'pattern': pattern,
                                              'size': size,
                                              'freq': results_all[sample][size][pattern]['freq'],
                                              'graphs': results_all[sample][size][pattern]['graphs']}
    request.session['graphs'] = json_dict


def build_tree(parent, local_hierarchy, local_dict, freqs):
    if isinstance(local_hierarchy, list):
        leaf_list = []
        for leaf in local_hierarchy:
            leaf_list.append({'text': leaf, 'tags': [str(freqs[leaf])]})
        return leaf_list
    else:
        nodes = []
        local_dict['nodes'] = []
        for child in local_hierarchy:
            nodes.append({'text': child, 'nodes': build_tree(child, local_hierarchy[child], local_dict['nodes'], freqs)})
        local_dict['text'] = parent
        local_dict['tags'] = [str(freqs[parent])]
        if len(nodes) == 0:
            del local_dict['nodes']
        else:
            local_dict['nodes'] = nodes
        return local_dict


def collect_hierarchy():
    freq_per_sample_pattern, patterns_hierarchy = hadoop.return_hierarchical_results()
    # create json like in bootstrap-treeview
    # [{text:, nodes:[]}, {...}]
    sample_dicts = {}
    for sample in patterns_hierarchy:
        json_dict = []
        for parent in patterns_hierarchy[sample]:
            json_dict_local = {}
            json_dict_local = (build_tree(parent, patterns_hierarchy[sample][parent], json_dict_local, freq_per_sample_pattern[sample]))
            json_dict.append(json_dict_local)
        sample_dicts[sample] = json_dict
    return sample_dicts


def collect_metrics(request):
    json_dict = {}
    results_all = hadoop.return_samples_metrics()
    for sample in results_all.keys():
        json_dict[sample] = {}
        for metric in results_all[sample].keys():
            json_dict[sample][metric] = results_all[sample][metric]
    request.session['sampleMetrics'] = json_dict


def collect_circos_data(request):
    url = static('textparser/GRCh37.json')
    request.session['chordsUrl'] = url
    urls_fusion = {}
    for sample in samples:
        urls_fusion[sample] = [static('textparser/' + sample + '_orig_for_circos.csv'),
                               static('textparser/' + sample + '_for_circos.csv')]
    request.session['fusionUrls'] = urls_fusion
    url_cyto = static('textparser/cytobands.csv')
    request.session['cytoUrl'] = url_cyto
    # temp for testing
    url_flare = static('textparser/flare.json')
    request.session['flareUrl'] = url_flare

hierarchy = collect_hierarchy()
print hierarchy