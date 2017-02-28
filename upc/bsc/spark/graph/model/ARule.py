import re
import csv
from StringIO import StringIO


class ARule:
    def __init__(self, line, size, calculate_abs=False):
        p = re.compile('\{(.*)\}')
        line = StringIO(line)
        fieldnames = []
        if calculate_abs:
            fieldnames=['1:nrow(p3)', 'numprecs', 'rules', 'support', 'confidence', 'lift',
                                                 'precedent', 'consequent', 'code']
        else:
            fieldnames=['1:nrow(p3)', 'numprecs', 'rules', 'support', 'confidence', 'lift',
                                                 'precedent', 'consequent', 'code', 'tid_full', 'tid_lhs']

        csv_reader = csv.DictReader(line, fieldnames)

        for row in csv_reader:
            self._conseq = p.search(row['consequent']).group(1) if  p.search(row['consequent']) != None else None
            self._ante = p.search(row['precedent']).group(1) if p.search(row['precedent']) != None else None
            self._support_rel = float(row['support'])
            self._conf = float(row['confidence'])
            self._lift = float(row["lift"])
            if(calculate_abs):
                self._support_abs = self._support_rel * float(size)
                self._support_abs_lhs = self._support_abs / self._conf
            else:
                # later add code for extracting absolute values
                self._support_abs = -1.0
                self._support_abs_lhs = -1.0
            self._full_tids = row['tid_full']
            self._lhs_tids = row['tid_lhs']



    @property
    def conseq(self):
        return self._conseq

    @conseq.setter
    def conseq(self, value):
        self._conseq = value

    @conseq.deleter
    def conseq(self):
        del self._conseq

    @property
    def ante(self):
        return self._ante

    @ante.setter
    def ante(self, value):
        self._ante = value

    @ante.deleter
    def ante(self):
        del self._ante

    @property
    def support_rel(self):
        return self._support_rel

    @support_rel.setter
    def support_rel(self, value):
        self._support_rel = value

    @support_rel.deleter
    def support_rel(self):
        del self._support_rel

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, value):
        self._conf = value

    @conf.deleter
    def conf(self):
        del self._conf

    @property
    def lift(self):
        return self._lift

    @lift.setter
    def lift(self, value):
        self._lift = value

    @lift.deleter
    def lift(self):
        del self._lift

    @property
    def support_abs(self):
        return self._support_abs

    @support_abs.setter
    def support_abs(self, value):
        self._support_abs = value

    @support_abs.deleter
    def support_abs(self):
        del self._support_abs

    @property
    def support_abs_lhs(self):
        return self._support_abs_lhs

    @support_abs_lhs.setter
    def support_abs_lhs(self, value):
        self._support_abs_lhs = value

    @support_abs_lhs.deleter
    def support_abs_lhs(self):
        del self._support_abs_lhs

    @property
    def full_tids(self):
        return self._full_tids

    @full_tids.setter
    def full_tids(self, value):
        self._full_tids = value

    @full_tids.deleter
    def full_tids(self):
        del self._full_tids

    @property
    def lhs_tids(self):
        return self._lhs_tids

    @lhs_tids.setter
    def lhs_tids(self, value):
        self._lhs_tids = value

    @lhs_tids.deleter
    def lhs_tids(self):
        del self._lhs_tids
