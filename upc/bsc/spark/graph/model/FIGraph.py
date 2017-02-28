

class FIGraph:
    #initial support is not defined, so -1
    def __init__(self, graph, count, sup_abs=-1.0, sup_rel=-1.0, full_tids=set(), lhs_tids=set()):
        self._fi_graph = graph
        self._count = count
        self._support_abs = sup_abs
        self._support_rel = sup_rel
        self._full_tids = full_tids
        self._lhs_tids = lhs_tids


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
    def support_rel(self):
        return self._support_rel

    @support_rel.setter
    def support_rel(self, value):
        self._support_rel = value

    @support_rel.deleter
    def support_rel(self):
        del self._support_rel

    @property
    def fi_graph(self):
        return self._fi_graph

    @fi_graph.setter
    def fi_graph(self, value):
        self._fi_graph = value

    @fi_graph.deleter
    def fi_graph(self):
        del self._fi_graph

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

    @count.deleter
    def count(self):
        del self._count

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
