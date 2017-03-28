

class SubgraphCollection:
    # initial support is not defined, so -1
    def __init__(self, label, subgraphs=list(), freq=0):
        self._label = label
        self._subgraphs = subgraphs
        self._freq = freq

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
    def subgraphs(self):
        return self._subgraphs

    @subgraphs.setter
    def subgraphs(self, value):
        self._subgraphs = value

    @subgraphs.deleter
    def subgraphs(self):
        del self._subgraphs

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        self._freq = value

    @freq.deleter
    def freq(self):
        del self._freq
