import numpy as np

from methods_collection.box_abstraction_monitoring.Abstractions import boxes_query


class Monitor(object):

    def __init__(self, class_y, good_ref=None, bad_ref=None):
        self.netName = "CIFAR10"
        self.classification = class_y
        self.location = "last"
        self.good_ref = good_ref
        self.bad_ref = bad_ref

    def set_reference(self, good_ref, bad_ref):
        self.good_ref = good_ref
        self.bad_ref = bad_ref

    def get_identity(self):
        print("Monitor for network:" + self.netName + "class: " + str(self.classification) + "at layer " + str(
            self.location))

    def make_verdicts(self, features):
        in_good_ref = []
        in_bad_ref = []
        if len(self.good_ref) and len(self.bad_ref):
            in_good_ref = ref_query(features, self.good_ref)
            in_bad_ref = ref_query(features, self.bad_ref)

        elif (not len(self.good_ref)) and len(self.bad_ref):
            in_good_ref = [False for x in features]
            in_bad_ref = ref_query(features, self.bad_ref)

        elif len(self.good_ref) and (not len(self.bad_ref)):
            in_good_ref = ref_query(features, self.good_ref)
            in_bad_ref = [False for x in features]

        else:
            in_good_ref = [False for x in features]
            in_bad_ref = [False for x in features]

        verdicts = query_infusion(in_good_ref, in_bad_ref)
        return verdicts


def ref_query(features, reference):
    query_results = [boxes_query(x, reference) for x in features]
    return query_results


def query_infusion(in_good_ref, in_bad_ref):
    if len(in_good_ref) == len(
            in_bad_ref):  # 0: acceptance (true, false), 1: rejection (false, true or false), 2: uncertainty (true, true)
        verdicts = np.zeros(len(in_good_ref), dtype=int)
        for i in range(len(in_good_ref)):
            if not in_good_ref[i]:
                verdicts[i] = 1
            elif in_bad_ref[i]:
                verdicts[i] = 2
        return verdicts
    else:
        print("Error: IllegalArgument")
