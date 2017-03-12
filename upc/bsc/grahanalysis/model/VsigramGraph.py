

class VsigramGraph:
    #initial support is not defined, so -1
    def __init__(self, graph, label='', sup_abs=-1.0, sup_rel=-1.0, edges=list(), vertices=list()):
        self._pn_graph = graph
        self._label = label
        self._support_abs = sup_abs
        self._support_rel = sup_rel
        self._edges = edges
        self._vertices = vertices


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
