from hdfs import Config
import ast
import csv


CHR_MAP = [249250621, 243199373, 198022430, 191154276, 180915260,
           171115067, 159138663, 146364022, 141213431, 135534747,
           135006516, 133851895, 115169878, 107349540, 102531392,
           90354753, 81195210, 78077248, 59128983, 63025520,
           48129895, 51304566, 155270560, 59373566]

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


def find_relative_position(position):
    # index = chr_number(chromosome)
    offset = 0L
    position = float(position)
    for i in range(len(CHR_MAP)):
        if offset < position < offset + long(CHR_MAP[i]):
            chrom_str = chr_str(i)
            pos = position - offset
            return chrom_str, pos
        else:
            offset += long(CHR_MAP[i])


def return_samples_metrics():
    client = Config().get_client('dev')
    # write the assignment file
    # with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv', 'rw') as csvfile:
    #     client.delete('sample', recursive=True)
    #     client.delete('samples/new_assignment_separate.csv', recursive=True)
    #     client.write('samples/new_assignment_separate.csv', csvfile)
    results_all = dict()
    samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
               'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
    for sample in samples:
        result_in_sample = dict()
        fname = 'samples/' + sample + '/' + sample + '_metrics.csv'
        with client.read(fname, encoding='utf-8') as reader:
            for line in reader:
                parts = line.split(',')
                metric_str = parts[0]
                val = parts[1]
                if metric_str != 'metric':
                    result_in_sample[metric_str] = val
        results_all[sample] = result_in_sample
        print results_all
    return results_all


def return_analysis_results():
    client = Config().get_client('dev')
    # write the assignment file
    # with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv', 'rw') as csvfile:
    #     client.delete('sample', recursive=True)
    #     client.delete('samples/new_assignment_separate.csv', recursive=True)
    #     client.write('samples/new_assignment_separate.csv', csvfile)
    results_all = dict()
    samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367',
               'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
    for sample in samples:
        result_in_sample = dict()
        for i in range(1, 4):
            dir_path = 'subgraphs/' + sample + '/' + str(i)
            fnames = client.list(dir_path)
            results = dict()
            for fname in fnames:
                with client.read(dir_path + '/' + fname, encoding='utf-8') as reader:
                    for line in reader:
                        parts = line.split(',', 1)
                        label_str = parts[0]
                        label_str = label_str[2:len(label_str) - 1].strip()
                        tuple_freq_list = parts[1].split(',', 1)
                        freq = int(tuple_freq_list[0][2:])
                        subgraphs_str = tuple_freq_list[1][0:len(tuple_freq_list[1]) - 3].strip()
                        subgraphs_list = ast.literal_eval(subgraphs_str)
                        # print subgraphs_list
                        subgraphs_list_for_circos = list()
                        for subgraph in subgraphs_list:
                            print subgraph
                            candidate_subgraph = []
                            for edge in subgraph:
                                print 'TYPE of edge: ' + str(type(edge))
                                chrom1, pos1 = find_relative_position(edge[0])
                                chrom2, pos2 = find_relative_position(edge[1])
                                candidate_edge = {'source_id': chrom1,
                                                  'source_breakpoint': pos1,
                                                  'target_id': chrom2,
                                                  'target_breakpoint': pos2,
                                                  'source_label': '',
                                                  'target_label': ''}
                                candidate_subgraph.append(candidate_edge)
                            subgraphs_list_for_circos.append(candidate_subgraph)

                        if label_str not in results:
                            results[label_str] = dict()
                            results[label_str]['freq'] = freq
                            results[label_str]['graphs'] = subgraphs_list_for_circos

                        else:
                            results[label_str]['freq'] += freq
                            results[label_str]['graphs'].extend(subgraphs_list_for_circos)
                result_in_sample[i] = results
        results_all[sample] = result_in_sample
        print results_all
    return results_all


return_analysis_results()