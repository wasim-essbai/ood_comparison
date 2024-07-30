from copy import deepcopy


class Box:
    def __init__(self):
         self.dimensions = None
         self.ivals = []

    def build(self, dimensions, points):
        piter = iter(points)
        self.dimensions = dimensions
        self.ivals = []
        try:
            point = next(piter)
        except StopIteration:
            return
        else:
            i = 0
            for coord in point:
                if i >= self.dimensions:
                    break
                self.ivals.append([coord, coord])
                i += 1
            if len(self.ivals) != self.dimensions:
                raise "IllegalArgument"

        while True:
            try:
                point = next(piter)
            except StopIteration:
                break
            else:
                i = 0
                for coord in point:
                    if i >= self.dimensions:
                        break
                    ival = self.ivals[i]
                    if coord < ival[0]:
                        ival[0] = coord
                    if coord > ival[1]:
                        ival[1] = coord
                    i += 1

    def query(self, point):
        i = 0
        for coord in point:
            if i >= self.dimensions:
                break
            ival = self.ivals[i]
            if coord < ival[0] or coord > ival[1]:
                return False
            i += 1
        return True

    def __str__(self):
        return self.ivals.__str__()


def boxes_query(point, boxes):
    for box in boxes:
        if len(box.ivals):
            if box.query(point):
                return True
