

class VsigramGraph:
    # initial support is not defined, so -1
    def __init__(self, graph, hash_str, label='', label_arr='', sup_abs=-1.0, sup_rel=-1.0,
                 edges=list(), vertices=list(), orig_edges=list(), orig_vertices=list()):
        self._pn_graph = graph
        self._label = label
        self._label_arr = label_arr
        self._support_abs = sup_abs
        self._support_rel = sup_rel
        self._edges = edges
        self._vertices = vertices
        self._orig_edges = orig_edges
        self._orig_vertices = orig_vertices
        self._hash_str = hash_str

    def __hash__(self):
        return hash(self.hash_str)

    def __eq__(self, other):
        # another object is equal to self, iff
        # it is an instance of MyClass
        return self.__hash__() == other.__hash__()

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
    def pn_graph(self):
        return self._pn_graph

    @pn_graph.setter
    def pn_graph(self, value):
        self._pn_graph = value

    @pn_graph.deleter
    def pn_graph(self):
        del self._pn_graph

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @label.deleter
    def label(self):
        del self._label

    @property
    def label_arr(self):
        return self._label_arr

    @label_arr.setter
    def label_arr(self, value):
        self._label_arr = value

    @label_arr.deleter
    def label_arr(self):
        del self._label_arr

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value

    @edges.deleter
    def edges(self):
        del self._edges

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    @vertices.deleter
    def vertices(self):
        del self._vertices

    @property
    def orig_edges(self):
        return self._orig_edges

    @orig_edges.setter
    def orig_edges(self, value):
        self._orig_edges = value

    @orig_edges.deleter
    def orig_edges(self):
        del self._orig_edges

    @property
    def orig_vertices(self):
        return self._orig_vertices

    @orig_vertices.setter
    def orig_vertices(self, value):
        self._orig_vertices = value

    @orig_vertices.deleter
    def orig_vertices(self):
        del self._orig_vertices

    @property
    def hash_str(self):
        return self._hash_str

    @hash_str.setter
    def hash_str(self, value):
        self._hash_str = value

    @hash_str.deleter
    def hash_str(self):
        del self._hash_str
