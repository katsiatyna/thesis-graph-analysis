

class ARGraph:
    # initial support is not defined, so -1
    def __init__(self, graph_lhs, graph_rhs, graph_full, count, sup_abs=-1.0, sup_rel=-1.0, lhs_tids=set(), full_tids=set()):
        self._graph_lhs = graph_lhs
        self._graph_rhs = graph_rhs
        self._graph_full = graph_full
        self._count = count
        self._support_abs = sup_abs
        self._support_rel = sup_rel
        self._full_tids = full_tids
        self._lhs_tids = lhs_tids
        self._conf = 0.0

    @property
    def graph_lhs(self):
        return self._graph_lhs

    @graph_lhs.setter
    def graph_lhs(self, value):
        self._graph_lhs = value

    @graph_lhs.deleter
    def graph_lhs(self):
        del self._graph_lhs

    @property
    def graph_rhs(self):
        return self._graph_rhs

    @graph_rhs.setter
    def graph_rhs(self, value):
        self._graph_rhs = value

    @graph_rhs.deleter
    def graph_rhs(self):
        del self._graph_rhs

    @property
    def graph_full(self):
        return self._graph_full

    @graph_full.setter
    def graph_full(self, value):
        self._graph_full = value

    @graph_full.deleter
    def graph_full(self):
        del self._graph_full

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
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

    @count.deleter
    def count(self):
        del self._count

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, value):
        self._conf = value

    @conf.deleter
    def conf(self):
        del self._conf
