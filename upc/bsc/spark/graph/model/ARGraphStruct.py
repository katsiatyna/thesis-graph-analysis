

class ARGraphStruct:
    def __init__(self, g_lhs,g_rhs, g_full):
        self._g_lhs = g_lhs
        self._g_rhs = g_rhs
        self._g_full = g_full

    @property
    def g_lhs(self):
        return self._g_lhs

    @g_lhs.setter
    def g_lhs(self, value):
        self._g_lhs = value

    @g_lhs.deleter
    def g_lhs(self):
        del self._g_lhs

    @property
    def g_rhs(self):
        return self._g_rhs

    @g_rhs.setter
    def g_rhs(self, value):
        self._g_rhs = value

    @g_rhs.deleter
    def g_rhs(self):
        del self._g_rhs

    @property
    def g_full(self):
        return self._g_full

    @g_full.setter
    def g_full(self, value):
        self._g_full = value

    @g_full.deleter
    def g_full(self):
        del self._g_full
