from functools import total_ordering


@total_ordering
class TClass(object):
    """
    The class is used to show detailed information about a class in a program.

    Attributes:
        @:param id:
        @:param name: str
        @:param attrdeps: dict
        @:param methdeps: dict
    """
    def __init__(self, id, name, attrdeps, methdeps):
        self.id = id
        self.name = name
        self.attrdeps = attrdeps
        self.methdeps = methdeps

    def printinfo(self):
        """
        Print attributes of an object.

        :return:
        """
        print("Id:", self.id)
        print("Name:", self.name)
        print("Attrbute Dependences:")
        for index,item in enumerate(self.attrdeps.items()):
            print("#"+str(index), " ", item)
        print("Method Dependences:")
        for index,item in enumerate(self.methdeps.items()):
            print("#"+str(index), " ", item)

    def getname(self):
        return self.name

    def getattrdeps(self):
        return self.attrdeps

    def getmethdeps(self):
        return self.methdeps

    def getid(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __gt__(self, other):
        return self.id > other.id

