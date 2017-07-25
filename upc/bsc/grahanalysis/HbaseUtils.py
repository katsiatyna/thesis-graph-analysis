import happybase
import pyhive


def recreate_sample_pattern_table():
    connection = happybase.Connection(autoconnect=False)
    connection.open()
    print(connection.tables())
    connection.disable_table('sample_pattern')
    connection.delete_table('sample_pattern')
    connection.create_table(
        'sample_pattern',
        {
            's': dict(max_versions=1),
            'p': dict(max_versions=1),
            'b': dict(max_versions=1),
            't': dict(max_versions=1),
            'e': dict(max_versions=1),
            'f': dict(max_versions=1)
        }
    )
    connection.close()


def recreate_sample_data_table():
    connection = happybase.Connection(autoconnect=False)
    connection.open()
    connection.disable_table('sample_data')
    connection.delete_table('sample_data')
    connection.create_table(
        'sample_data',
        {
            's': dict(max_versions=1),
            'b': dict(max_versions=1),
            't': dict(max_versions=1),
            'd': dict(max_versions=1),
            'm': dict(max_versions=1)

        }
    )
    connection.close()


def recreate_sample_metrics_table():
    connection = happybase.Connection(autoconnect=False)
    connection.open()
    connection.disable_table('sample_metrics')
    connection.delete_table('sample_metrics')
    connection.create_table(
        'sample_metrics',
        {
            's': dict(max_versions=1),
            'b': dict(max_versions=1),
            't': dict(max_versions=1),
            'm': dict(max_versions=1)
        }
    )
    connection.close()


def read_data():
    connection = happybase.Connection(autoconnect=False)
    connection.open()
    sample_pattern_table = connection.table('sample_pattern')
    for key, data in sample_pattern_table.scan():
        #print key
        sample = data[b's:name']
        cancer = data[b's:cancer']
        bandwidth = data[b'b:value']
        threshold_counter = data[b't:counter']
        threshold = data[b't:value']
        pattern = data[b'p:code']
        size = data[b'p:size']
        # list of tuples of strings
        inter_embeddings = [(inter_key, inter_value) for inter_key, inter_value in data.iteritems() if inter_key.startswith(b'e:inter_')]
        inter_edges = [map(float, val[0].split('_')[1].split(':')) for val in inter_embeddings]
        #print inter_edges
        intra_embeddings = [(intra_key, intra_value) for intra_key, intra_value in data.iteritems() if intra_key.startswith(b'e:intra_')]
        intra_edges = [map(float, val[0].split('_')[1].split(':')) for val in intra_embeddings]
        #print intra_edges
        print sample, cancer, bandwidth, threshold_counter, threshold, pattern, size
    connection.close()


def read_sample_data():
    connection = happybase.Connection(autoconnect=False)
    connection.open()
    sample_pattern_table = connection.table('sample_data')
    for key, data in sample_pattern_table.scan():
        print key
        sample = data[b's:name']
        cancer = data[b's:cancer']
        bandwidth = data[b'b:value']
        threshold_counter = data[b't:counter']
        threshold = data[b't:value']
        metrics = [(m_key, m_value) for m_key, m_value in data.iteritems() if m_key.startswith(b'm:')]
        chr1_list = data[b'd:chr1']
        pos1_list = data[b'd:pos1']
        chr2_list = data[b'd:chr2']
        pos2_list = data[b'd:pos2']
        # list of tuples of strings

        print sample, cancer, bandwidth, threshold_counter, threshold, metrics, type(chr1_list), type(pos1_list), type(chr2_list), type(pos2_list)
    connection.close()


#recreate_sample_pattern_table()
# read_data()
#recreate_sample_data_table()
#read_sample_data()
