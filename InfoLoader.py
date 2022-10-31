import os
import re
import csv
from TClass import TClass
from TEdge import TEdge
from RType import RType


class InfoLoader(object):
    """The class is used to load data coming from csv file into the environment of Deep Reinforce Learning .

    Parameters
    ----------
    path: `str`
        A string showing absolute path of selected program folder in folder, infoAbtPg
    mode: `int`
        An integer showing the mode of class InfoLoader
    ncls: `int`
        An integer showing the number of classes in selected program
    id2name: `dict`
        A dictionary mapping from the id of a class to the corresponding name of the class
    routes: `list`
        A 2-dimensional list showing connectivity relationships between classes
    clpmat: `list`
        A 2-dimensional list showing coupling relationships between classes
    tedges: `list`
        A list containing all TEdge objects.
    id2cls: `dict`
        A dictionary mapping from the id of a class to the corresponding TClass object of the class
    """
    def __init__(self, infopath, mode):
        self.path = infopath
        self.mode = mode
        self.ncls = self.getClassNum()
        self.id2name = self.getId2NameDict()
        self.routes = self.getRoutes()
        self.clpmat = self.getCoulpingMatrix()
        self.edges = self.getTEdges()
        self.id2cls = self.getId2TClassDict()

    def getClassNum(self):
        """
        Obtain the number of all classes in selected program

        :return: num: int
        """
        csv_path = os.path.join(self.path, "CId_Name.csv")
        num = 0
        with open(csv_path, "r") as f:
            num = len(f.readlines()) - 1
        return num

    def getId2NameDict(self):
        """
        Obtain a dictionary mapping from {integer} to {string}

        :return dic :
        :rtype dict
        """
        csv_path = os.path.join(self.path, "CId_Name.csv")
        ids =[]
        names =[]
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            reader.__next__()
            for items in reader:
                ids.append(int(items[0]))
                names.append(items[1])
            f.close()
        dic = dict(zip(ids, names))
        # print("\n".join(dic.values()))
        return dic

    def getToClassName2AttrdepNumDict(self, id):
        """
        Obtain a dictionary mapping from {string} to {dictionary}

        :param id: int
        :return: dic: dict
        """
        csv_path = os.path.join(self.path, "Attr_Method_deps.csv")
        names = []
        attrdeps_num = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            temp = []
            for items in reader:
                if int(items[0]) == id:
                    temp = items
                    break
            attrdep = temp[2]
            if attrdep != "null":
                split_attrdep = re.split(r'{|}|,', attrdep)
                for index, part in enumerate(split_attrdep):
                    if index != 1 and len(part) != 0:
                        splits = part.split(':')
                        names.append(splits[0])
                        attrdeps_num.append(int(splits[1]))
        dic = dict(zip(names, attrdeps_num))
        return dic

    def getToClassName2MethdepNumDict(self, id):
        """
        Obatin a dictionary mapping from {string} to {dictionary}

        :param id: int
        :return: dic: dict
        """
        csv_path = os.path.join(self.path, "Attr_Method_deps.csv")
        names = []
        methdeps_num = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            temp = []
            for items in reader:
                if int(items[0]) == id:
                    temp = items
                    break
            methdep = temp[3]
            if methdep != "null":
                split_methdep = re.split(r'{|}|,', methdep)
                for index, part in enumerate(split_methdep):
                    if index != 1 and len(part) != 0:
                        splits = part.split(':')
                        names.append(splits[0])
                        methdeps_num.append(int(splits[1]))
        dic = dict(zip(names, methdeps_num))
        return dic

    def getId2ImportanceDict(self):
        """
        Obtain a dictionary mapping from {integer} to {float}

        :return: dic: dict
        """
        ids = []
        imps = []
        csv_path = os.path.join(self.path, "CId_importance.csv")
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            for items in reader:
                id = int(items[0])
                imp = float(items[1])
                ids.append(id)
                imps.append(imp)
        dic = dict(zip(ids, imps))
        return dic

    def strtype2RType(self, strtype):
        if strtype == 'NONE':
            return RType.NONE
        elif strtype == 'As':
            return RType.AS
        elif strtype == 'Ag':
            return RType.AG
        elif strtype == 'I':
            return RType.I
        elif strtype == 'Dy':
            return RType.DY
        return None

    def getTEdges(self):
        tedges = []
        csv_path = os.path.join(self.path, "deps_type.csv")
        id2tclass_dict = self.getId2TClassDict()
        # print(id2tclass_dict)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            if self.mode:
                for items in reader:
                    etype = self.strtype2RType(items[3])
                    from_tclass = id2tclass_dict[int(items[0])]
                    to_tclass = id2tclass_dict[int(items[1])]
                    tedge = TEdge(from_tclass, to_tclass, etype)
                    tedges.append(tedge)
                    # tedge.printinfo()
            else:
                for items in reader:
                    # print(items)
                    etype = self.strtype2RType(items[3])
                    # print(etype)
                    if etype != RType.DY:
                        from_tclass = id2tclass_dict[int(items[0])]
                        to_tclass = id2tclass_dict[int(items[1])]
                        tedge = TEdge(from_tclass, to_tclass, etype)
                        tedges.append(tedge)
                        # tedge.printinfo()
            f.close()
        return tedges

    def getId2TClassDict(self):
        """
        Obtain a dictionary mapping from {id} to {TClass}

        :return: dic: dict
        """
        tclasses = []
        ids = []
        id2name_dict = self.getId2NameDict()
        for id, name in id2name_dict.items():
            attrdeps = self.getToClassName2AttrdepNumDict(id)
            methdeps = self.getToClassName2MethdepNumDict(id)
            tclass = TClass(id, name, attrdeps, methdeps)

            ids.append(id)
            tclasses.append(tclass)

        dic = dict(zip(ids, tclasses))
        return dic

    def getRoutes(self):
        csv_path = os.path.join(self.path, "deps_type.csv")
        routes = [[0] * self.getClassNum() for _ in range(self.getClassNum())]
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            if self.mode:
                for items in reader:
                    fromid = int(items[0])
                    toid = int(items[1])
                    typenum = int(items[2])
                    routes[fromid][toid] = typenum
            else:
                for items in reader:
                    typenum = int(items[2])
                    if typenum != 4:
                        fromid = int(items[0])
                        toid = int(items[1])
                        routes[fromid][toid] = typenum
            f.close()
        return routes

    def getCoulpingMatrix(self):
        csv_path = os.path.join(self.path, "Couple_List.csv")
        clpmat = [[0] * self.getClassNum() for _ in range(self.getClassNum())]
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            for items in reader:
                fromid = int(items[0])
                toid = int(items[1])
                clpmat[fromid][toid] = float(items[2])
            f.close()
        return clpmat


if __name__ == "__main__":
    loader = InfoLoader(".\\infoAbtPg\\ant", 0)
    print(loader.getClassNum())
    loader.getId2NameDict()
    print(loader.getToClassName2AttrdepNumDict(5))
    print(loader.getToClassName2AttrdepNumDict(11))
    print(loader.getToClassName2AttrdepNumDict(24))
    print()
    print(loader.getToClassName2MethdepNumDict(5))
    print(loader.getToClassName2MethdepNumDict(11))
    print(loader.getToClassName2MethdepNumDict(24))

    for key, val in loader.getId2TClassDict().items():
        print(key)
        val.printinfo()
        print()

    print(loader.getId2ImportanceDict())
    print()
    tedges = loader.getTEdges()
    tedges = sorted(tedges)
    for tedge in tedges:
        tedge.printinfo()

    print(loader.getRoutes())
    print(loader.getCoulpingMatrix())







