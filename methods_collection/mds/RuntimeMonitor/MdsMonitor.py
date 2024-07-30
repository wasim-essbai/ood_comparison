from methods_collection.mds.RuntimeMonitor.mds_utils import *


class MdsMonitor(object):
    def __init__(self, noise, num_classes, feature_type_list, reduce_dim_list, device):
        self.magnitude = noise
        self.feature_type_list = feature_type_list
        self.reduce_dim_list = reduce_dim_list

        self.num_classes = num_classes
        self.num_layer = len(self.feature_type_list)

        self.feature_mean = None
        self.feature_precision = None
        self.transform_matrix = None
        self.alpha_list = [1]

        self.setup_flag = False
        self.device = device

    def setup(self, model, set_loader, out_set_loader=None):
        if not self.setup_flag:
            # step 1: estimate initial mean and variance from training set
            self.feature_mean, self.feature_precision, self.transform_matrix = get_MDS_stat(model,
                                                                                            set_loader,
                                                                                            self.num_classes,
                                                                                            self.feature_type_list,
                                                                                            self.reduce_dim_list,
                                                                                            self.device)
            # step 2: input process and hyperparam searching for alpha
            if self.alpha_list is None:
                print('\n Searching for optimal alpha list...')
                # get in-distribution scores
                for layer_index in range(self.num_layer):
                    M_in = get_Mahalanobis_scores(
                        model, set_loader, self.num_classes,
                        self.feature_mean, self.feature_precision,
                        self.transform_matrix, layer_index,
                        self.feature_type_list, self.magnitude, self.device)
                    M_in = np.asarray(M_in, dtype=np.float32)
                    if layer_index == 0:
                        Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                    else:
                        Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
                # get out-of-distribution scores
                for layer_index in range(self.num_layer):
                    M_out = get_Mahalanobis_scores(
                        model, out_set_loader, self.num_classes,
                        self.feature_mean, self.feature_precision,
                        self.transform_matrix, layer_index,
                        self.feature_type_list, self.magnitude, self.device)
                    M_out = np.asarray(M_out, dtype=np.float32)
                    if layer_index == 0:
                        Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                    else:
                        Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
                Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
                Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

                # logistic regression for optimal alpha
                self.alpha_list = alpha_selector(Mahalanobis_in,
                                                 Mahalanobis_out)
            self.setup_flag = True
        else:
            pass

    def process_input(self, model, data):
        for layer_index in range(self.num_layer):
            pred, score = compute_Mahalanobis_score(model,
                                                    data,
                                                    self.num_classes,
                                                    self.feature_mean,
                                                    self.feature_precision,
                                                    self.transform_matrix,
                                                    layer_index,
                                                    self.feature_type_list,
                                                    self.magnitude,
                                                    self.device,
                                                    return_pred=True)
            if layer_index == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)
        alpha = torch.FloatTensor(self.alpha_list).to(self.device)
        conf = torch.matmul(score_list, alpha)
        return pred, conf
