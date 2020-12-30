""" """

from dalitzphsp import DalitzPhsp

class C3Alpha(DalitzPhsp):
    def __init__(self, mmo, mres, malpha=3.72738):
        super().__init__(mmo, malpha, malpha, malpha)
        self.mres = mres
