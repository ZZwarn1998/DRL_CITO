import math
from functools import total_ordering


@total_ordering
class Datum(object):
    """
    A class used to store an output during training process

    Attributes:
        :param order: list
            Generated integration test order
        :param ocplx: float
            Overall complexity of stubs
        :param appear: int
            the time when the result was generated and whose unit of measurement is millisecond
        :param GStubs: int
            the number of generic stubs
        :param SStubs: int
            the number of special stubs
        :param deps: int
            overall dependency
        :param attrdeps: int
            attribute dependency
        :param methdeps: int
            method dependency
        :param infoGStubs: dict
            a dictionary which contains {string} as key and {string} as value to show detailed information about generic
             stubs
        :param infoSStubs: dict
            a dictionary which contains {string} as key and {string} as value to show detailed information about special
             stubs
    """
    def __init__(self, order=None, ocplx=None, appear=None, GStubs=None, SStubs=None,
              deps=None, attrdeps=None, methdeps=None, infoGStubs=None, infoSStubs=None):
        """
        Constructor of class Datum

        :param order: list
        :param ocplx: float
        :param appear: int
        :param GStubs: int
        :param SStubs: int
        :param deps: int
        :param attrdeps: int
        :param methdeps: int
        :param infoGStubs: dict
        :param infoSStubs: dict
        :return:
        """
        self.order = order
        self.ocplx = ocplx
        self.appear = appear
        self.GStubs = GStubs
        self.SStubs = SStubs
        self.deps = deps
        self.attrdeps = attrdeps
        self.methdeps = methdeps
        self.infoGStubs = infoGStubs
        self.infoSStubs = infoSStubs

    # def setDatum(self, order=None, ocplx=None, appear=None, GStubs=None, SStubs=None,
    #           deps=None, attrdeps=None, methdeps=None, infoGStubs=None, infoSStubs=None):
    #     self.order = order
    #     self.ocplx = ocplx
    #     self.appear = appear
    #     self.GStubs = GStubs
    #     self.SStubs = SStubs
    #     self.deps = deps
    #     self.attrdeps = attrdeps
    #     self.methdeps = methdeps
    #     self.infoGStubs = infoGStubs
    #     self.infoSStubs = infoSStubs

    def __lt__(self,other):
        if math.isclose(self.ocplx, other.ocplx):
            return self.ocplx > other.ocplx
        else:
            return self.SStubs > self.SStubs

    def __eq__(self, other):
        return self.ocplx == self.ocplx