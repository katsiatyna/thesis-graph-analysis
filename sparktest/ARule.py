import re
import csv
from StringIO import StringIO

class ARule:
    def __init__(self, line):
        p = re.compile('\{(.*)\}')
        line = StringIO(line)
        csv_reader = csv.DictReader(line, fieldnames=['1:nrow(p3)', 'numprecs', 'rules', 'support', 'confidence', 'lift',
                                                 'precedent', 'consequent', 'code'])
        for row in csv_reader:
            self.conseq = p.search(row['consequent']).group(1) if  p.search(row['consequent']) != None else None
            self.ante = p.search(row['precedent']).group(1) if p.search(row['precedent']) != None else None
