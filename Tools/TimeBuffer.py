class TimeBuffer():

    def __init__(self, size, initialVals = None):
        self.data = [initialVals] * size
        self.size = size

    def __index(self, time):
        return time % self.size

    def get(self, time):
        return self.data[self.__index(time)]

    def insert(self, time, val):
        self.data[self.__index(time)] = val

    def getData(self):
        return self.data