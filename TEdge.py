from TClass import TClass
from functools import total_ordering


@total_ordering
class TEdge(object):
    def __init__(self, fromtclass, totclass, etype):
        self.fromtclass = fromtclass
        self.totclass = totclass
        self.type = etype
        self.fromid = fromtclass.getid()
        self.toid = totclass.getid()

    def printinfo(self):
        print(self.fromtclass.getname(), self.fromid, "->", self.totclass.getname(), self.toid, self.type.name)

    def getfromclass(self):
        return self.fromtclass

    def gettoclass(self):
        return self.totclass

    def gettype(self):
        return self.type

    def getfromid(self):
        return self.fromid

    def gettoid(self):
        return self.toid

    def __eq__(self, other):
        return self.fromid == other.fromid \
               and self.toid == other.toid \
               and self.type == other.type

    def __gt__(self, other):
        if self.fromid != other.fromid:
            return self.fromid > other.fromid
        if self.toid != other.toid:
            return self.toid > other.toid
        if self.type != other.type:
            return self.type.value > other.type.value




