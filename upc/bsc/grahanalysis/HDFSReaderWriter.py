from hdfs import Config
import ast
import csv

client = Config().get_client('dev')
# write the assignment file
with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment_separate.csv', 'rw') as csvfile:
    client.delete('sample', recursive=True)
    client.delete('samples/new_assignment_separate.csv', recursive=True)
    client.write('samples/new_assignment_separate.csv', csvfile)
samples = ['7d734d06-f2b1-4924-a201-620ac8084c49', '0448206f-3ade-4087-b1a9-4fb2d14e1367', 'ea1cac20-88c1-4257-9cdb-d2890eb2e123']
for sample in samples:
    file_name = sample + '_new_assignment.csv'
    metrics_file_name = sample + '_metrics.csv'
    with open('/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/' + file_name,
              'rw') as csvfile:
        client.delete('samples/' + sample + '/' + file_name, recursive=True)
        client.write('samples/' + sample + '/' + file_name, csvfile)
    with open('/home/kkrasnas/Documents/thesis/pattern_mining/candidates/' + sample + '/' + metrics_file_name,
              'rw') as csvfile:
        client.delete('samples/' + sample + '/' + metrics_file_name, recursive=True)
        client.write('samples/' + sample + '/' + metrics_file_name, csvfile)
results_all = dict()
for i in range(1, 4):
    dir_path = 'subgraphs/ea1cac20-88c1-4257-9cdb-d2890eb2e123/' + str(i)
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
                if label_str not in results:
                    results[label_str] = freq
                else:
                    results[label_str] += freq
        results_all[i] = results
print results_all


