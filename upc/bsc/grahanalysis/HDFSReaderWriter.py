from hdfs import Config
import ast
import csv

client = Config().get_client('dev')
# write the assignment file
with open('/home/kkrasnas/Documents/thesis/pattern_mining/validation_data/new_assignment.csv', 'rw') as csvfile:
    client.delete('sample', recursive=True)
    client.delete('samples/new_assignment_separate2.csv', recursive=True)
    client.write('samples/new_assignment_separate.csv', csvfile)
for i in range(1, 4):
    dir_path = 'subgraphs/7d734d06-f2b1-4924-a201-620ac8084c49/' + str(i)
    fnames = client.list(dir_path)
    results = []
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
                results.append((label_str, (freq, subgraphs_list)))
                print results

