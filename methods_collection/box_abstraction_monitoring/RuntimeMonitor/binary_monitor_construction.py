import numpy as np
import pickle
import os
from methods_collection.box_abstraction_monitoring.RuntimeMonitor import BinaryMonitor
from itertools import combinations

def binary_monitor_online_generation(features_arrary, y_features, y_pred, gamma, path_to_store, num_classes):
    labels_set = set(np.arange(0, num_classes).tolist())
    print(os.getcwd())
    for y in labels_set:
        print(f'Building monitor for class {y}')
        x_feat = features_arrary[y_pred == y, :]
        y_labels = y_features[y_pred == y]
        indices_correct_predictions = y_labels == 1

        good_features = x_feat[indices_correct_predictions, :]
        
        monitor_y_i = BinaryMonitor(0)
        monitor_y_i.initialize(good_features.shape[1])
        prog = 0
        for feat in good_features:
            monitor_y_i.add(feat)
            for pos in combinations(range(good_features.shape[1]), gamma):
                    new_feat = []
                    for i in range(good_features.shape[1]):
                        if i in pos:
                            new_feat.append(1-feat[i])
                        else:
                            new_feat.append(feat[i])
                    monitor_y_i.add(new_feat)
            prog +=1
            print('\r' + ' Training: ' + str(round(100 * prog/good_features.shape[0], 2)) + '% complete..', end ="")

        monitor_stored_path = f'{path_to_store}monitor_{y}.pkl'
        with open(monitor_stored_path, 'wb') as f:
            pickle.dump(monitor_y_i, f)
