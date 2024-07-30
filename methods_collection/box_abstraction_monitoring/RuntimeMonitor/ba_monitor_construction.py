import numpy as np
import pickle
from sklearn.cluster import KMeans
from methods_collection.box_abstraction_monitoring.Abstractions import Box
from methods_collection.box_abstraction_monitoring.RuntimeMonitor import Monitor
import os


def modified_kmeans_cluster(values_to_cluster, threshold, k_start, n_clusters=None):
    if n_clusters is not None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(values_to_cluster)
        return kmeans.labels_
    else:
        n_clusters = k_start
        n_values = len(values_to_cluster)
        assert n_values > 0
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(values_to_cluster)
        inertias = [kmeans.inertia_]
        while n_values > n_clusters:
            n_clusters_new = n_clusters + 1
            kmeans_new = KMeans(n_clusters=n_clusters_new, random_state=0).fit(values_to_cluster)
            inertias.append(kmeans_new.inertia_)
            if terminate_clustering(inertias, threshold):
                break
            kmeans = kmeans_new
            n_clusters += 1
        return kmeans.labels_


def terminate_clustering(inertias, threshold):
    # method: compute relative improvement toward previous step
    assert len(inertias) > 1
    improvement = 1 - (inertias[-1] / inertias[-2])
    return improvement < threshold


# tau: the clustering parameter; k_and_taus：dictionary storing the number of clusters for existing clustering parameters
def start_k(tau, k_and_taus):
    taus_existed = [float(key) for key in k_and_taus.keys()]
    k_start = 1
    if len(taus_existed):
        bigger_taus = [x for x in taus_existed if x > tau]
        if len(bigger_taus):
            tau_closest = min(bigger_taus)
            k_start = k_and_taus[str(tau_closest)]
    return k_start


def create_local_abstractions(values_to_cluster, tau, k_start):
    n_dim = values_to_cluster.shape[1]  # the vector dimension

    # cluster the vectors and return cluster labels for each vector
    cluster_labels = modified_kmeans_cluster(values_to_cluster, tau, k_start)
    num_clusters = max(cluster_labels) + 1
    labels = range(num_clusters)
    # taus_existed.append(tau)
    # k_and_taus[str(tau)] = max(cluster_labels) + 1

    # extract the indices of vectors in a cluster
    clusters_indices = []
    for k in labels:
        indices_cluster_k, = np.where(cluster_labels == k)
        clusters_indices.append(indices_cluster_k)

    # creat local box for each cluster
    loc_boxes = [Box() for i in labels]
    for j in labels:
        loc_boxes[j].build(n_dim, values_to_cluster[clusters_indices[j]])

    return num_clusters, loc_boxes


def monitor_online_generation(features_arrary, y_features, y_pred, tau, path_to_store, num_classes):
    labels_set = set(np.arange(0, num_classes).tolist())
    print(os.getcwd())
    for y in labels_set:
        print(f'Building monitor for class {y}')
        x_feat = features_arrary[y_pred == y, :]
        y_labels = y_features[y_pred == y]
        indices_correct_predictions = y_labels == 1
        indices_incorrect_predictions = y_labels == 0

        good_features = x_feat[indices_correct_predictions, :]
        bad_features = x_feat[indices_incorrect_predictions, :]

        k_and_taus_good = dict()
        k_and_taus_bad = dict()

        if len(bad_features) == 0:
            bad_loc_boxes = []
        else:
            # partition features and build a list of boxes
            k_start_bad = start_k(tau, k_and_taus_bad)
            k_new_bad, bad_loc_boxes = create_local_abstractions(bad_features, tau, k_start_bad)
            k_and_taus_bad[str(tau)] = k_new_bad  # store the number of clusters for next tau

        if len(good_features) == 0:
            good_loc_boxes = []
        else:
            k_start_good = start_k(tau, k_and_taus_good)
            k_new_good, good_loc_boxes = create_local_abstractions(good_features, tau, k_start_good)
            k_and_taus_good[str(tau)] = k_new_good

        # build a monitor at layer i for class y under the setting of clustering parameter tau
        # save the built monitor as well
        monitor_y_i = Monitor(y, good_ref=good_loc_boxes, bad_ref=bad_loc_boxes)

        monitor_stored_path = f'{path_to_store}monitor_{y}.pkl'
        with open(monitor_stored_path, 'wb') as f:
            pickle.dump(monitor_y_i, f)
