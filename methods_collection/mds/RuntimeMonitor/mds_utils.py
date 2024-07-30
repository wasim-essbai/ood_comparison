import torch
import sklearn.covariance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from methods_collection.mds.RuntimeMonitor.InverseLDA import *
from torch.autograd import Variable


def tensor2list(x):
    return x.data.cpu().tolist()


def get_torch_feature_stat(feature, only_mean=False):
    feature = feature.view([feature.size(0), feature.size(1), -1])
    feature_mean = torch.mean(feature, dim=-1)
    feature_var = torch.var(feature, dim=-1)
    if feature.size(-2) * feature.size(-1) == 1 or only_mean:
        # [N, C, 1, 1] does not need variance for kernel
        feature_stat = feature_mean
    else:
        feature_stat = torch.cat((feature_mean, feature_var), 1)
    return feature_stat


def process_feature_type(feature_temp, feature_type):
    if feature_type == 'flat':
        feature_temp = feature_temp.view([feature_temp.size(0), -1])
    elif feature_type == 'stat':
        feature_temp = get_torch_feature_stat(feature_temp)
    elif feature_type == 'mean':
        feature_temp = get_torch_feature_stat(feature_temp, only_mean=True)
    else:
        raise ValueError('Unknown feature type')
    return feature_temp


def reduce_feature_dim(feature_list_full, label_list_full, feature_process):
    if feature_process == 'none':
        transform_matrix = np.eye(feature_list_full.shape[1])
    else:
        feature_process, kept_dim = feature_process.split('_')
        kept_dim = int(kept_dim)
        if feature_process == 'capca':
            lda = InverseLDA(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        elif feature_process == 'pca':
            pca = PCA(n_components=kept_dim)
            pca.fit(feature_list_full)
            transform_matrix = pca.components_.T
        elif feature_process == 'lda':
            lda = LinearDiscriminantAnalysis(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        else:
            raise Exception('Unknown Process Type')
    return transform_matrix


def alpha_selector(data_in, data_out):
    label_in = np.ones(len(data_in))
    label_out = np.zeros(len(data_out))
    data = np.concatenate([data_in, data_out])
    label = np.concatenate([label_in, label_out])
    # skip the last-layer flattened feature (duplicated with the last feature)
    lr = LogisticRegressionCV(n_jobs=-1).fit(data, label)
    alpha_list = lr.coef_.reshape(-1)
    print(f'Optimal Alpha List: {alpha_list}')
    return alpha_list


def get_Mahalanobis_scores(model, test_loader, num_classes,
                           sample_mean, precision, transform_matrix,
                           layer_index, feature_type_list, magnitude, device):
    model.eval()
    Mahalanobis = []
    for data, target in test_loader:
        data = data.to(device)
        data = Variable(data, requires_grad=True)
        noise_gaussian_score = compute_Mahalanobis_score(
            model, data, num_classes, sample_mean, precision, transform_matrix,
            layer_index, feature_type_list, magnitude, device)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
    return Mahalanobis


@torch.no_grad()
def get_MDS_stat(model, set_loader, num_classes, feature_type_list, reduce_dim_list, device):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    model.eval()
    num_layer = len(feature_type_list)
    feature_class = [[None for x in range(num_classes)]
                     for y in range(num_layer)]
    feature_all = [None for x in range(num_layer)]
    label_list = []
    # collect features
    for data, target in set_loader:
        data = data.to(device)
        target = target.to(device)
        _, feature_list = model(data, return_feature_list=True)
        label_list.extend(tensor2list(target))
        for layer_idx in range(num_layer):
            feature_type = feature_type_list[layer_idx]
            feature_processed = process_feature_type(feature_list[layer_idx], feature_type)
            if isinstance(feature_all[layer_idx], type(None)):
                feature_all[layer_idx] = tensor2list(feature_processed)
            else:
                feature_all[layer_idx].extend(tensor2list(feature_processed))
    label_list = np.array(label_list)
    # reduce feature dim and split by classes
    transform_matrix_list = []
    for layer_idx in range(num_layer):
        feature_sub = np.array(feature_all[layer_idx])
        transform_matrix = reduce_feature_dim(feature_sub, label_list,
                                              reduce_dim_list[layer_idx])
        transform_matrix_list.append(torch.Tensor(transform_matrix).to(device))
        feature_sub = np.dot(feature_sub, transform_matrix)
        for feature, label in zip(feature_sub, label_list):
            feature = feature.reshape([-1, len(feature)])
            if isinstance(feature_class[layer_idx][label], type(None)):
                feature_class[layer_idx][label] = feature
            else:
                feature_class[layer_idx][label] = np.concatenate(
                    (feature_class[layer_idx][label], feature), axis=0)
    # calculate feature mean
    feature_mean_list = [[
        np.mean(feature_by_class, axis=0)
        for feature_by_class in feature_by_layer
    ] for feature_by_layer in feature_class]

    # calculate precision
    precision_list = []
    for layer in range(num_layer):
        X = []
        for k in range(num_classes):
            X.append(feature_class[layer][k] - feature_mean_list[layer][k])
        X = np.concatenate(X, axis=0)
        # find inverse
        group_lasso.fit(X)
        precision = group_lasso.precision_
        precision_list.append(precision)

    # put mean and precision to cuda
    feature_mean_list = [torch.Tensor(i).to(device) for i in feature_mean_list]
    precision_list = [torch.Tensor(p).to(device) for p in precision_list]

    return feature_mean_list, precision_list, transform_matrix_list


def compute_Mahalanobis_score(model,
                              data,
                              num_classes,
                              sample_mean,
                              precision,
                              transform_matrix,
                              layer_index,
                              feature_type_list,
                              magnitude,
                              device,
                              return_pred=False):
    # extract features
    data.requires_grad = True
    _, out_features = model(data, return_feature_list=True)
    out_features = process_feature_type(out_features[layer_index], feature_type_list[layer_index])
    out_features = torch.mm(out_features, transform_matrix[layer_index])

    # compute Mahalanobis score
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

    # Input_processing
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # here we use the default value of 0.5
    gradient.index_copy_(
        1,
        torch.LongTensor([0]).to(device),
        gradient.index_select(1, torch.LongTensor([0]).to(device)) / 0.5)
    gradient.index_copy_(
        1,
        torch.LongTensor([1]).to(device),
        gradient.index_select(1, torch.LongTensor([1]).to(device)) / 0.5)
    gradient.index_copy_(
        1,
        torch.LongTensor([2]).to(device),
        gradient.index_select(1, torch.LongTensor([2]).to(device)) / 0.5)
    tempInputs = torch.add(data.data, gradient, alpha=-magnitude)  # updated input data with perturbation

    with torch.no_grad():
        _, noise_out_features = model(Variable(tempInputs), return_feature_list=True)
        noise_out_features = process_feature_type(noise_out_features[layer_index], feature_type_list[layer_index])
        noise_out_features = torch.mm(noise_out_features, transform_matrix[layer_index])

    noise_gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
    if return_pred:
        return sample_pred, noise_gaussian_score
    else:
        return noise_gaussian_score
