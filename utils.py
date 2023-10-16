import math
import matplotlib.pyplot as plt
import numpy
import scipy


# Functions for visualizing data

def plot_scatter(x_val, y_val, L, axisLabels, classNames, name, save=False):
    """
    Scatter plot that distinguish between the two classes of a given binary task
    :param x_val: array of x values for the scatter plot
    :param y_val: array of y values for the scatter plot
    :param L: array of labels of the samples
    :param axisLabels: list of the two labels for the axis
    :param classNames: list of the two classNames (..[i]=className of class i)
    :param name: figure will be saved at path: name.png
    :param save: set to true if you want to permanently save the plot
    """
    plt.figure()
    plt.xlabel(axisLabels[0])
    plt.ylabel(axisLabels[1])
    plt.scatter(x_val[L == 0], y_val[L == 0], label=classNames[0], alpha=0.5)
    plt.scatter(x_val[L == 1], y_val[L == 1], label=classNames[1], alpha=0.5)
    plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
    plt.legend()
    if save:
        plt.savefig(name + ".png")
    plt.show()


def plot_hist(D, L, name="", save=False):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    print(D0.shape)
    print(D1.shape)

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel("Feature " + str(dIdx))
        plt.hist(D0[dIdx, :], bins=30, density=True, alpha=0.4, label='Spoofed')
        plt.hist(D1[dIdx, :], bins=30, density=True, alpha=0.4, label='Authentic')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        if save:
            plt.savefig(name + str(dIdx) + ".png")
    plt.show()


def print_dimension_ranges(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for dIdx in range(D.shape[0]):
        D0_F_Max = numpy.max(D0[dIdx, :])
        D0_F_Min = numpy.min(D0[dIdx, :])
        D1_F_Max = numpy.max(D1[dIdx, :])
        D1_F_Min = numpy.min(D1[dIdx, :])
        print("Feature " + str(dIdx) + " Class:Male -> range:" + str(D0_F_Max - D0_F_Min))
        print("Feature " + str(dIdx) + " Class:Female -> range:" + str(D1_F_Max - D1_F_Min))
    pass


def plot_heatmap(D, title, save=False):
    correlation_matrix = numpy.corrcoef(D)
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Features')
    step_size = 1
    plt.xticks(numpy.arange(0, 10, step_size))
    plt.yticks(numpy.arange(0, 10, step_size))
    if save:
        plt.savefig(title + ".pdf")
    plt.show()


def compute_features_statistics(D, L):
    """
    Compute some features statistics of the given dataset (for a binary task)
    :param D: Dataset of columns vectors
    :param L: Labels of samples
    :return: [D0_means, D1_means, D0_stds, D1_stds]
    """
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D0_means = []
    D1_means = []
    D0_stds = []
    D1_stds = []
    for dIdx in range(D.shape[0]):
        D0_means.append(numpy.mean(D0[dIdx, :]))
        D1_means.append(numpy.mean(D1[dIdx, :]))
        D0_stds.append(numpy.std(D0[dIdx, :]))
        D1_stds.append(numpy.std(D1[dIdx, :]))
    return [D0_means, D1_means, D0_stds, D1_stds]

# ----------------------------------------------------------------------------------------------------------------------

# Base functions for managing data

def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        '0': 0,
        '1': 1,
    }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()  # strip() to remove the newline at the end
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:  # try except in case a line generate an exception (e.g. last line is a newline)
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def split_db_k_to_1(D, L, k, seed=0):
    """
    Split the given database in k folds, without making attention at generating folds that are similar to the starting
    dataset (in terms of balance of classes).
    :param D: Dataset of columns vectors (numpy array with shape (n_features, n_samples)
    :param L: Numpy array of labels
    :param k: Number of folds
    :param seed: Seed used for the random permutation of values when creating the k-folds
    :return: k_folds (list of folds that are numpy arrays with shape (n_features, n_samples_per_fold)), label_k_fold(
    list of numpy arrays containing labels corresponding to each element in k_folds)
    """
    k_folds = []
    label_k_folds = []
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxSplit = numpy.array_split(idx, k)
    for i in range(k):
        k_folds.append(D[:, idxSplit[i]])
        label_k_folds.append(L[idxSplit[i]])

    return k_folds, label_k_folds


def split_db_k_to_1_v2(D, L, k, seed=0):
    """
        Split the given database in k folds for a binary task, making attention at generating folds that are similar to
        the starting dataset (in terms of balance of classes).

        It does not work if "k" is greater than N
        where N = number of samples of the minority class.
        :param D: Dataset of columns vectors (numpy array with shape (n_features, n_samples)
        :param L: Numpy array of labels
        :param k: Number of folds
        :param seed: Seed used for the random permutation of values when creating the k-folds
        :return: k_folds (list of folds that are numpy arrays with shape (n_features, n_samples_per_fold)), label_k_fold(
        list of numpy arrays containing labels corresponding to each element in k_folds)
        """
    k_folds = []
    label_k_folds = []
    numpy.random.seed(seed)
    DTC = D[:, L == 1]
    DFC = D[:, L == 0]
    LTC = L[L == 1]
    LFC = L[L == 0]
    # print("DTC:" + str(DTC.shape))
    # print("DFC:" + str(DFC.shape))
    # print("LTC:"+str(LTC.shape))
    # print("LFC:"+str(LFC.shape))
    idx_TC = numpy.random.permutation(DTC.shape[1])
    idx_FC = numpy.random.permutation(DFC.shape[1])

    idx_TC_split = numpy.array_split(idx_TC, k)
    idx_FC_split = numpy.array_split(idx_FC, k)
    for i in range(k):
        k_fold_TC = DTC[:, idx_TC_split[i]]
        k_fold_FC = DFC[:, idx_FC_split[i]]
        # print("k_fold_TC:"+str(k_fold_TC.shape))
        # print("k_fold_FC:" + str(k_fold_FC.shape))
        k_folds.append(numpy.hstack([k_fold_TC, k_fold_FC]))
        label_k_folds.append(numpy.concatenate([LTC[idx_TC_split[i]], LFC[idx_FC_split[i]]]))
    # print(k_folds)
    # print()
    # print(label_k_folds)
    return k_folds, label_k_folds


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# ----------------------------------------------------------------------------------------------------------------------

# Functions for preprocessing data

def center_Dataset(D):
    mu = D.mean(1)
    DC = D - vcol(mu)
    return DC


def calculate_pca_proj_matrix(D, m):
    """
    Returns the projection matrix (P) that must be used to project the entire dataset in the PCA space (numpy.dot(P.T, D)).

    The function does not center the dataset before looking for the m directions with highest variance.

    Advice: center the datasets before applying PCA, to avoid a predominant direction towards the dataset mean.
    :param D: Dataset
    :param m: Number of dimensions of the PCA space
    :return: Projection matrix P
    """
    C = numpy.dot(D, D.T) / D.shape[1]
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P


def LDA2(D, L, m):
    Sb = compute_Sb(D, L)
    Sw = compute_Sw(D, L)
    # U, s, _ = numpy.linalg.svd(Sw)
    s, U = numpy.linalg.eigh(Sw)
    P1 = numpy.dot(U * vrow(1.0 / (s ** 0.5)), U.T)
    Sbt = numpy.dot(numpy.dot(P1, Sb), P1.T)
    # U, _, _ = numpy.linalg.svd(Sbt)
    s, U = numpy.linalg.eigh(Sbt)
    P2 = U[:, ::-1][:, 0:m]
    # P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)
    return W


def standardize_variances(D):
    """
    Preprocessing technique that transforms features of dataset so that they have unit variance
    :param D: Dataset to transform (numpy array with shape (n_features, n_samples)
    :return: Transformed dataset
    """
    for i in range(D.shape[0]):
        std_dev = numpy.std(D[i, :])
        D[i, :] = D[i, :] / std_dev
    return D


def z_normalization(D):
    """
    Preprocessing technique that transforms features of dataset so that they have unit variance and zero mean.

    It does not modify the input dataset.
    :param D: Dataset to transform (numpy array with shape (n_features, n_samples)
    :return: Transformed dataset
    """
    Dtmp = numpy.zeros_like(D)
    for i in range(D.shape[0]):
        std_dev = numpy.std(D[i, :])
        mean = numpy.mean(D[i, :])
        Dtmp[i, :] = (D[i, :] - mean) / std_dev
    return Dtmp


def L2_normalization(D):
    """
    Preprocessing technique that transforms features of dataset so that they have unit Euclidean norm
    :param D:
    :return:
    """
    for i in range(D.shape[0]):
        norm_x = numpy.linalg.norm(D[i, :], ord=2)
        D[i, :] = D[i, :] / norm_x
    return D


def min_max_scaling(D):
    """
    Preprocessing technique that transforms values of the features so that they are in a range between 0 and 1
    :param D:
    :return:
    """
    for i in range(D.shape[0]):
        min_val = numpy.min(D[i, :])
        max_val = numpy.max(D[i, :])
        D[i, :] = (D[i, :] - min_val) / (max_val - min_val)
    return D


# This is probably not correct
def whiten_cov_matrix(D, L):
    Sw = compute_Sw(D, L)
    U, s, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot(U * vrow(1.0 / (s ** 0.5)), U.T)
    for i in range(D.shape[1]):
        D[:, i] = numpy.dot(P1, D[:, i])
    return D


def compute_Sw(D, L):  # where D is dataset matrix and k is the number of classes
    C = 0
    for i in range(L.max() + 1):
        DC = center_Dataset(D[:, L == i])
        n_c = numpy.size(DC, 1)
        C = C + float(n_c) * numpy.dot(DC, DC.T) / DC.shape[1]
    return C / float(D.shape[1])


def compute_Sb(D, L):  # where D is dataset matrix and k is the number of classes
    mu = vcol(D.mean(1))
    C = 0
    for i in range(L.max() + 1):
        mu_i = vcol(D[:, L == i].mean(1))
        n_c = numpy.size(D[:, L == i], 1)
        C = C + float(n_c) * numpy.dot(mu_i - mu, (mu_i - mu).T)
    return C / float(D.shape[1])

# ---------------------------------------------------------------------------------------------

# Functions for Gaussian models

def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * math.log(2 * math.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - vcol(mu)
    return firstTerm - 0.5 * (XC * numpy.dot(L, XC)).sum(0)


def calculate_mu_C(D, naive_bayes=False):
    mu = vcol(D.mean(1))
    DC = D - vcol(mu)
    C = numpy.dot(DC, DC.T) / DC.shape[1]
    if naive_bayes:
        C = C * numpy.identity(C.shape[1])
    return mu, C


def generate_log_score_matrix(DTR, LTR, DTE, naive_bayes=False, tied_covariance=False):
    class_samples = []
    class_parameters = []
    row_list = []
    class_num = LTR.max() + 1
    for i in range(class_num):
        class_samples.append(DTR[:, LTR == i])
        class_parameters.append(calculate_mu_C(class_samples[i], naive_bayes))
        if not tied_covariance:
            row_list.append((logpdf_GAU_ND(DTE, *class_parameters[i])))

    if tied_covariance:
        Ctied = numpy.zeros(class_samples[0].shape[0])
        for i in range(class_num):
            Ctied = Ctied + class_samples[i].shape[1] * class_parameters[i][1]
        Ctied = Ctied / DTR.shape[1]
        for i in range(class_num):
            row_list.append(logpdf_GAU_ND(DTE, (class_parameters[i])[0], Ctied))

    return numpy.vstack(row_list)


def generate_joint_log_density(SM, P):
    return SM + numpy.log(P)


def generate_gaussian_scores(DTR, LTR, DTE, naive_bayes=False, tied_covariance=False):
    """
    Compute the output scores (llrs) of a Gaussian model for a binary task.

    N.B: It computes llrs(scores), not class posterior probabilities, it generates only application-independent scores.
    :param DTR: Training dataset of column feature vectors
    :param LTR: Labels of training dataset as numpy array with shape (n_trainingSamples,)
    :param DTE: numpy array with shape (n_features, n_evalSamples) representing dataset on which output scores are computed
    :param naive_bayes: set to true to use a naive-bayes approach
    :param tied_covariance: set to true to use a tied-covariance approach
    :return: numpy array of output scores (llrs) with shape (n_evalSamples,)
    """
    Score_Matrix = generate_log_score_matrix(DTR, LTR, DTE, naive_bayes, tied_covariance)
    S = Score_Matrix[1, :] - Score_Matrix[0, :]  # subtract because they are logarithms
    return S


def k_fold_validation_gaussian(D, L, k, split_function):
    """
    It performs k-fold cross validation for a binary task using 4 different gaussian models (MVG, NB, Tied_MVG, Tied_NB)
    :param D: Dataset of columns vectors, numpy array with shape (n_features, n_samples)
    :param L: Correct labels, numpy array with shape (n_samples,)
    :param k: Number of folds for the cross validation
    :param split_function: function used for splitting the database (must return list of k_folds and list of label_k_folds)
    :return: S_MVG, S_NB, S_tied_MVG, S_tied_NB (numpy arrays of scores(llrs) for each different Gaussian model), L (numpy array of labels)

    """
    k_folds, label_k_folds = split_function(D, L, k)
    S_MVG = []
    S_NB = []
    S_tied_MVG = []
    S_tied_NB = []
    for i in range(k):
        DTE = k_folds.pop(i)
        DTR = numpy.hstack(k_folds)
        LTE = label_k_folds.pop(i)
        LTR = numpy.hstack(label_k_folds)

        # Multivariate Gaussian Model
        S = generate_gaussian_scores(DTR, LTR, DTE, naive_bayes=False, tied_covariance=False)
        S_MVG.append(S)

        # Naive Bayes Gaussian Model
        S = generate_gaussian_scores(DTR, LTR, DTE, naive_bayes=True, tied_covariance=False)
        S_NB.append(S)

        # Tied Covariance Model
        S = generate_gaussian_scores(DTR, LTR, DTE, naive_bayes=False, tied_covariance=True)
        S_tied_MVG.append(S)

        # Tied Naive Bayes
        S = generate_gaussian_scores(DTR, LTR, DTE, naive_bayes=True, tied_covariance=True)
        S_tied_NB.append(S)

        k_folds.insert(i, DTE)
        label_k_folds.insert(i, LTE)

    return numpy.hstack(S_MVG), numpy.hstack(S_NB), numpy.hstack(S_tied_MVG), numpy.hstack(S_tied_NB), numpy.hstack(
        label_k_folds)

# ---------------------------------------------------------------------------------------------

# Functions for logistic regression models

def to_expanded_feature_space(D):
    """
    Generate a mapping fi for the Dataset, so that has quadratic separation surfaces in the original space,
    and linear separation surfaces in the space defined by the mapping fi.

    This implementation uses only the upper triangular matrix of vec(xx'), since it's symmetric, in this way we have
    few numbers of features, thus the LR model is easier to train.
    :param D: Dataset, numpy array of column feature vectors (with shape (n_features, n_samples))
    :return: Expanded feature space.
    """
    n = ((D.shape[0] ** 2) - D.shape[0]) / 2 + D.shape[
        0]  # I take only the superior triangle of the matrix, since x*x.T is symmetric
    l = []
    for i in range(D.shape[1]):
        x = vcol(D[:, i])
        vec_xxT = vcol(numpy.ravel(numpy.dot(x, x.T))[0:int(n)])
        fi_x = numpy.concatenate([vec_xxT, x])
        l.append(fi_x)
    return numpy.hstack(l)


def logreg_obj_wrap(DTR, LTR, l):
    dim = DTR.shape[0]
    ZTR = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = vcol(v[0:dim])
        b = v[-1]
        scores = numpy.dot(w.T, DTR) + b
        loss_per_sample = numpy.logaddexp(0, -ZTR * scores)
        loss = loss_per_sample.mean() + 0.5 * l * numpy.linalg.norm(w) ** 2
        return loss

    return logreg_obj


def compute_w_b(DTR, LTR, l):
    """
    Compute model parameters w and b for the regularized binary logistic regression model given a particular value of lambda.
    It exploits the empirical prior of the training set.
    :param DTR: Training dataset of column feature vectors
    :param LTR: Labels of training dataset as numpy array with shape (n_trainingSamples,)
    :param l: lambda (for regularized LR model)
    :return: LR model parameters w, b
    """
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    x0 = numpy.zeros(DTR.shape[0] + 1)
    v = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    w, b = v[0][0:-1], v[0][-1]
    return w, b


def logreg_obj_wrap_with_priors(DTR, LTR, l, P_t):
    dim = DTR.shape[0]

    def logreg_obj(v):
        w = vcol(v[0:dim])
        b = v[-1]
        scores = numpy.dot(w.T, DTR) + b
        scores_T = scores[:, LTR == 1]
        scores_F = scores[:, LTR == 0]

        loss_per_sample_T = P_t * numpy.logaddexp(0, -scores_T)
        loss_per_sample_F = (1 - P_t) * numpy.logaddexp(0, scores_F)
        loss = loss_per_sample_T.mean() + loss_per_sample_F.mean() + 0.5 * l * numpy.linalg.norm(w) ** 2
        return loss

    return logreg_obj


def compute_w_b_with_priors(DTR, LTR, l, P_t):
    """
    Compute model parameters w and b for the regularized binary logistic regression model given a particular value of lambda.
    It generates a model that embeds the prior P_t instead of the empirical prior of the training set
    :param DTR: Training dataset of column feature vectors
    :param LTR: Labels of training dataset as numpy array with shape (n_trainingSamples,)
    :param l: lambda (for regularized LR model)
    :param P_t: Prior probability of class True (class 1)
    :return: LR model parameters w, b
    """
    logreg_obj = logreg_obj_wrap_with_priors(DTR, LTR, l, P_t)
    x0 = numpy.zeros(DTR.shape[0] + 1)
    v = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    w, b = v[0][0:-1], v[0][-1]
    return w, b


def compute_logreg_scores_with_priors(DTR, LTR, DTE, l, P_t):
    """
    Compute the output scores (llrs) of a prior weighted regularized logistic regression model for a binary task,
    embedding the prior P_t in the model (instead of the empirical prior of the dataset)
    :param DTR: Training dataset of column feature vectors
    :param LTR: Labels of training dataset as numpy array with shape (n_trainingSamples,)
    :param DTE: Evaluation dataset on which the output scores are computed
    :param l: lambda (for regularized LR model)
    :param P_t: Prior probability of class True (class 1)
    :return: numpy array of scores with shape (n_trainingSamples,)
    """
    w, b = compute_w_b_with_priors(DTR, LTR, l, P_t)  # compute a model with different empirical priors
    S = numpy.dot(w.T, DTE) + b
    return S


def k_fold_validation_logreg(D, L, k, l, P_t, split_function):
    """
        Perform k-fold cross validation for a binary task using a logistic regression model with priors embedded in the model.
        :param D: Dataset of columns vectors, numpy array with shape (n_features, n_samples)
        :param L: Correct labels, numpy array with shape (n_samples,)
        :param k: Number of folds for the cross validation
        :param l: Hyper parameter lambda used in for regularizing logreg model
        :param P_t: Prior probability of class true (class 1)
        :param split_function: function used for splitting the database (must return list of k_folds and list of label_k_folds)
        :return: S_f (numpy array of scores(llrs) generated from the logreg model using k-fold validation), L (numpy array of labels)
        """
    k_folds, label_k_folds = split_function(D, L, k)
    S_f = []

    for i in range(k):
        DTE = k_folds.pop(i)
        DTR = numpy.hstack(k_folds)
        LTE = label_k_folds.pop(i)
        LTR = numpy.hstack(label_k_folds)
        # Logistic Regression
        S = compute_logreg_scores_with_priors(DTR, LTR, DTE, l, P_t)
        S_f.append(S)

        k_folds.insert(i, DTE)
        label_k_folds.insert(i, LTE)

    return numpy.hstack(S_f), numpy.hstack(label_k_folds)

# ---------------------------------------------------------------------------------------------

# Functions for SVM

def k_fold_validation_SVM(D, L, k, C, p_t, split_function, kernel_function):
    """
    Perform k-fold cross validation for a binary task using a soft-margin SVM model with priors embedded in the model.
    :param D: Dataset of columns vectors, numpy array with shape (n_features, n_samples)
    :param L: Correct labels, numpy array with shape (n_samples,)
    :param k: Number of folds for the cross validation
    :param C: Hyper parameter C used in for soft-margin SVM model
    :param p_t: prior probability of class true for the target application
    :param split_function: function used for splitting the database (must return list of k_folds and list of label_k_folds)
    :param kernel_function: kernel function used for computing the kernel matrix
    :return: S_f (numpy array of scores(llrs) generated from the SVM model using k-fold validation), L (numpy array of labels)
    """
    k_folds, label_k_folds = split_function(D, L, k)
    S_f = []
    p_t_emp = L[L == 1].shape[0] / L.shape[0]

    for i in range(k):
        DTE = k_folds.pop(i)
        DTR = numpy.hstack(k_folds)
        LTE = label_k_folds.pop(i)
        LTR = numpy.hstack(label_k_folds)

        # SVM
        S = compute_SVM_scores(DTR, LTR, DTE, C * p_t / p_t_emp, C * (1 - p_t) / (1 - p_t_emp), kernel_function)
        S_f.append(S)

        k_folds.insert(i, DTE)
        label_k_folds.insert(i, LTE)
    return numpy.hstack(S_f), numpy.hstack(label_k_folds)


def compute_H(DTR, LTE, k):
    """
    Function that computes the matrix H of the soft-margin SVM dual problem using the given kernel function k
    :param DTR: numpy array of shape (d, n) containing the training data
    :param LTE: numpy array of shape (n,) containing the training labels
    :param k: kernel function that takes as input two datasets D1, D2 and returns a numpy array of shape (n, n) with the kernel matrix computed between samples of D1 and D2
    :return: numpy array of shape (n, n) containing the matrix H
    """
    G = k(DTR, DTR)
    ZTR = LTE * 2.0 - 1.0  # ZTR is a vector of shape (n,) containing only 1 and -1
    H = numpy.outer(ZTR, ZTR) * G
    return H


def SVM_dual_obj_wrap(DTR, LTR, k):
    H = compute_H(DTR, LTR, k)

    def SVM_dual_obj(alpha):
        """
        Function that, given a numpy array alpha, computes the soft-margin SVM dual objective and its gradient
        :param alpha: numpy array of alphas
        :return: SVM dual objective and its gradient as a tuple evaluated at alpha
        """

        J1 = 0.5 * numpy.dot(numpy.dot(alpha.transpose(), H), alpha) - numpy.dot(alpha.transpose(), vcol(numpy.ones(H.shape[1])))
        grad_J = numpy.dot(H, alpha) - numpy.ones_like(alpha)
        return J1, grad_J

    return SVM_dual_obj


def compute_dual_solution(DTR, LTR, C_t, C_f, k):
    """
    Function that computes the dual solution of the soft-margin SVM problem with different hyper-parameters C_t and C_f for class true anc class false, on dataset DTR
    :param DTR: numpy array of shape (d, n) containing the training data
    :param LTR: numpy array of shape (n,) containing the training labels
    :param C_t: soft-margin SVM hyperparameter C for the positive class
    :param C_f: soft-margin SVM hyperparameter C for the negative class
    :param k: kernel function that takes as input two datasets D1, D2 and returns a numpy array of shape (n, n) with the kernel matrix computed between samples of D1 and D2
    :return: numpy array of shape (n,) containing the dual solution
    """
    SVM_dual_obj = SVM_dual_obj_wrap(DTR, LTR, k)
    x0 = numpy.zeros(DTR.shape[1])
    v, _, _ = scipy.optimize.fmin_l_bfgs_b(SVM_dual_obj, x0, bounds=[(0, C_t) if LTR[i] == 1 else (0, C_f) for i in range(DTR.shape[1])], factr=10000)
    alpha = v
    return alpha


def compute_SVM_scores(DTR, LTR, DTE, C_t, C_f, kernel_function):
    """
    Function that, given a training set (DTR, LTR) and a test set DTE, computes the soft-margin SVM scores on DTE, using the given kernel_function to solve
    the dual problem, with different hyper-parameters C_t and C_f for class true anc class false

    The underlying functions need the kernel function to output k(x1, x2) + K^2, so that also the bias term is regularized, and thus we can solve the dual problem
    using the L-BFGS-B solver.

    K can be thought as a hyperparameter that controls the regularization of the bias term.
    :param DTR: numpy array of shape (d, n) containing the training data
    :param LTR: numpy array of shape (n,) containing the training labels
    :param DTE: numpy array of shape (d, m) containing the test data
    :param C_t: soft-margin SVM hyperparameter for the positive class
    :param C_f: soft-margin SVM hyperparameter for the negative class
    :param kernel_function: kernel function that takes as input two datasets D1, D2 and returns a numpy array of shape (n, n) with the kernel matrix computed between samples of D1 and D2
    """
    ZTR = LTR * 2.0 - 1.0  # ZTR is a vector of shape (n,) containing only 1 and -1
    alpha = compute_dual_solution(DTR, LTR, C_t, C_f, kernel_function)
    SVM_scores = numpy.dot(alpha * ZTR, kernel_function(DTR, DTE))
    return SVM_scores


def kernel_poly(d, c):
    """
    Function that returns a polynomial kernel function with degree d and constant c
    :param d: degree of the polynomial kernel
    :param c: constant of the polynomial kernel
    :return: polynomial kernel function
    """

    def kernel(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** d

    return kernel


def kernel_RBF(gamma):
    """
    Function that returns a RBF kernel function with parameter gamma
    :param gamma: parameter of the RBF kernel
    :return: RBF kernel function
    """

    def kernel(DTR, DTE):
        return numpy.exp(-gamma * (numpy.sum(DTR ** 2, axis=0)[:, numpy.newaxis] + numpy.sum(DTE ** 2, axis=0) - 2 * numpy.dot(DTR.T, DTE)))

    return kernel

# ---------------------------------------------------------------------------------------------

# Functions for Gaussian mixture models

def logpdf_GMM(X, gmm):
    """
    It computes the log-density of a GMM for a set of samples contained in matrix X.
    :param X: matrix of samples with shape (D,N) where D is the dimensionality of the samples and N the number of samples.
    :param gmm: a GMM encoding, in the form of [(w1,mu1,C1),...,(wK,muK,CK)]. Each tuple contains the parameters of a Gaussian component.
    """
    S = generate_joint_log_densities(X, gmm)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens


def generate_joint_log_densities(X, gmm):
    """
    It computes the joint log-density of a GMM for a set of samples contained in matrix X.
    :param X: matrix of samples with shape (D,N) where D is the dimensionality of the samples and N the number of samples.
    :param gmm: a GMM encoding, in the form of [(w1,mu1,C1),...,(wK,muK,CK)]. Each tuple contains the parameters of a Gaussian component.
    :return: a matrix S with shape (K,N) where S[k,n] is the log-density of the k-th Gaussian component for the n-th sample.
    """
    S = numpy.zeros((len(gmm), X.shape[1]))
    for g in range(len(gmm)):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
        S[g, :] += numpy.log(gmm[g][0])
    return S


def generate_class_posterior_probabilities_from_joint_log(SJoint_log):
    """
    It computes the class posterior probabilities (responsibilities in case of GMM) from the joint log-densities.
    :param SJoint_log: a matrix S with shape (K,N) where S[k,n] is the log-density of the k-th Gaussian component for the n-th sample.
    """
    S_log_marginal = vrow(scipy.special.logsumexp(SJoint_log, axis=0))
    logSPost = SJoint_log - S_log_marginal
    return numpy.exp(logSPost)


def GMM_generate_log_score_matrix(DTR, LTR, DTE, G, psi, alpha=0.1, eps=1e-6, diagonal=[False, False], tied_covariance=[False, False]):
    row_list = []
    class_num = LTR.max() + 1
    for i in range(class_num):
        mu, C = calculate_mu_C(DTR[:, LTR == i])
        if diagonal[i]:
            C = C * numpy.eye(C.shape[0])
        gmm = EM_LBG(DTR[:, LTR == i], [(1.0, mu, C)], G[i], psi, alpha, eps, diagonal[i], tied_covariance[i])
        row_list.append(logpdf_GMM(DTE, gmm))
    return numpy.vstack(row_list)


def generate_GMM_scores(DTR, LTR, DTE, G, psi, alpha=0.1, eps=1e-6, diagonal=[False, False], tied_covariance=[False, False]):
    """
    It computes the log-likelihood ratio scores for a binary task using a GMM model
    :param DTR: matrix of training samples with shape (D,N) where D is the dimensionality of the samples and N the number of samples.
    :param LTR: vector of training class labels with shape (N,).
    :param DTE: matrix of test samples with shape (D,M) where D is the dimensionality of the samples and M the number of samples.
    :param G: vector of number of Gaussian components.(G[0]=n for class 0, G[1]=n for class 1)
    :param psi: minimum allowed covariance eigenvalue (to avoid degenerate solution with covariance matrices that shrink to zero)
    :param alpha: hyperparameter used for split in the LBG algorithm
    :param eps: convergence threshold used in the LBG algorithm
    :param diagonal: if True, the covariance matrices are forced to be diagonal
    :param tied_covariance: if True, the covariance matrices are forced to be equal inside each single GMM
    """
    Score_Matrix = GMM_generate_log_score_matrix(DTR, LTR, DTE, G, psi, alpha, eps, diagonal, tied_covariance)
    S = Score_Matrix[1, :] - Score_Matrix[0, :]  # subtract because they are logarithms
    return S


def k_fold_validation_GMM(D, L, k, split_function, G, psi, alpha=0.1, eps=1e-6, diagonal=[False, False], tied_covariance=[False, False]):
    """
    It performs k-fold cross validation for a binary task using a GMM model
    :param D: matrix of samples with shape (D,N) where D is the dimensionality of the samples and N the number of samples.
    :param L: vector of class labels with shape (N,).
    :param k: number of folds.
    :param split_function: function that splits the data into k folds.
    :param G: vector of number of Gaussian components.(G[0]=n for class 0, G[1]=n for class 1)
    :param psi: minimum allowed covariance eigenvalue (to avoid degenerate solution with covariance matrices that shrink to zero)
    :param alpha: hyperparameter used for split in the LBG algorithm
    :param eps: minimum log-likelihood increase allowed in the EM algorithm
    :param diagonal: if True, the covariance matrices are forced to be diagonal
    :param tied_covariance: if True, the covariance matrices are forced to be equal inside each single GMM
    :return S_GMM: vector of llr scores for the evaluation samples
    :return label_k_folds: vector of class labels for the evaluation samples
    """
    k_folds, label_k_folds = split_function(D, L, k)
    S_GMM = []

    for i in range(k):
        DTE = k_folds.pop(i)
        DTR = numpy.hstack(k_folds)
        LTE = label_k_folds.pop(i)
        LTR = numpy.hstack(label_k_folds)
        S = generate_GMM_scores(DTR, LTR, DTE, G, psi, alpha, eps, diagonal, tied_covariance)
        S_GMM.append(S)

        k_folds.insert(i, DTE)
        label_k_folds.insert(i, LTE)

    return numpy.hstack(S_GMM), numpy.hstack(label_k_folds)


def GMM_generate_joint_log_density(SM, P):
    return SM + numpy.log(P)


def compute_accuracy(SPost, correctLabels):
    predictedLabels = numpy.argmax(SPost, 0)
    res = predictedLabels == correctLabels
    correctPredictions = res.sum(0)
    acc = correctPredictions / res.size
    err = 1 - acc
    return acc, err


def generate_model_parameters(X, SPost, psi, diagonal=False, tied_covariance=False):
    """
    It computes the parameters of a GMM from a set of samples contained in matrix X and the class posterior probabilities (responsibilities).

    """
    D, N = X.shape
    K = SPost.shape[0]
    gmm = []
    for g in range(K):
        Zg = numpy.sum(SPost[g, :])
        Fg = numpy.dot(SPost[g, :], X.T)
        S_g = numpy.dot(SPost[g, :] * X, X.T)
        w_g = Zg / N
        mu_g = Fg / Zg
        C_g = S_g / Zg - numpy.dot(vcol(mu_g), vrow(mu_g))
        if diagonal:
            C_g = C_g * numpy.eye(C_g.shape[0])
        gmm.append((w_g, mu_g, C_g))
    if tied_covariance:
        C = numpy.zeros((D, D))
        for g in range(K):
            C += gmm[g][0] * gmm[g][2]
        for g in range(K):
            gmm[g] = (gmm[g][0], gmm[g][1], C)
    return gmm


def compute_average_log_likelihood(X, gmm):
    """
    It computes the average log-likelihood of a GMM for a set of samples contained in matrix X.
    :param X: matrix of samples with shape (D,N) where D is the dimensionality of the samples and N the number of samples.
    :param gmm: a GMM encoding, in the form of [(w1,mu1,C1),...,(wK,muK,CK)]. Each tuple contains the parameters of a Gaussian component.
    """
    logdens = logpdf_GMM(X, gmm)
    return numpy.mean(logdens)


def min_eigenvalue_constraint(C, psi):
    U, s, _ = numpy.linalg.svd(C)
    s[s < psi] = psi
    return numpy.dot(U, vcol(s) * U.T)


def EM_estimation(X, gmm_init, psi, eps=1e-6, diagonal=False, tied_covariance=False):
    """
    It estimates a GMM from a set of samples contained in matrix X.

    It constrains the eigenvalues of the covariance matrices to be larger or equal to a lower bound psi, in order to avoid degenerate solutions, where
    one or more covariance matrices of components shrink to zero.
    :param X: matrix of samples with shape (D,N) where D is the dimensionality of the samples and N the number of samples.
    :param gmm_init: a GMM encoding, in the form of [(w1,mu1,C1),...,(wK,muK,CK)]. Each tuple contains the parameters of a Gaussian component.
    :param psi: a lower bound for the eigenvalues of the covariance matrices.
    :param eps: a small value used as a stopping criterion for the EM algorithm.
    :return: a GMM encoding, in the form of [(w1,mu1,C1),...,(wK,muK,CK)]. Each tuple contains the parameters of a Gaussian component.
    """
    gmm = gmm_init

    logdens_old = compute_average_log_likelihood(X, gmm)
    while True:
        S = generate_joint_log_densities(X, gmm)
        SPost = generate_class_posterior_probabilities_from_joint_log(S)
        gmm = generate_model_parameters(X, SPost, psi, diagonal=diagonal, tied_covariance=tied_covariance)
        for i in range(len(gmm)):
            gmm[i] = (gmm[i][0], gmm[i][1], min_eigenvalue_constraint(gmm[i][2], psi))
        logdens = compute_average_log_likelihood(X, gmm)

        if logdens - logdens_old < 0:
            print("Log-likelihood decreased!")
            print(logdens - logdens_old)
        if numpy.abs(logdens - logdens_old) < eps:
            break
        logdens_old = logdens
    return gmm


def G_split(gmm, alpha=0.1):
    """
    Split a GMM with G components into 2G components.

    The split is performed along the direction of the largest variance, using a step that is proportional to the standard deviation
    of the component we are splitting
    :param gmm: a GMM encoding with G components
    :return: a new GMM encoding with 2G components
    """
    gmm_new = []
    for gmm_comp in gmm:
        U, s, Vh = numpy.linalg.svd(gmm_comp[2])
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmm_comp1 = (gmm_comp[0] / 2, vcol(gmm_comp[1]) + vcol(d), gmm_comp[2])
        gmm_comp2 = (gmm_comp[0] / 2, vcol(gmm_comp[1]) - vcol(d), gmm_comp[2])
        gmm_new.append(gmm_comp1)
        gmm_new.append(gmm_comp2)
    return gmm_new


def EM_LBG(X, gmm_init, G, psi, alpha=0.1, eps=1e-6, diagonal=False, tied_covariance=False):
    """

    """
    gmm = gmm_init
    gmm[0] = (1, gmm[0][1], min_eigenvalue_constraint(gmm[0][2], psi))
    while len(gmm) < G:
        gmm = G_split(gmm, alpha)
        gmm = EM_estimation(X, gmm, psi, eps=eps, diagonal=diagonal, tied_covariance=tied_covariance)
    return gmm[::-1]


# ---------------------------------------------------------------------------------------------


# Functions for evaluation of models

def compute_accuracy(predictedLabels, correctLabels):
    """
    Compute the accuracy given an array of predicted and an array of correct labels
    :param predictedLabels: numpy array of predicted labels with shape (n,)
    :param correctLabels: numpy array of correct labels with shape (n,)
    :return: accuracy, error computed over the two arrays of labels
    """
    res = predictedLabels == correctLabels
    correctPredictions = res.sum(0)
    acc = correctPredictions / res.size
    err = 1 - acc
    return acc, err


def compute_confusion_matrix(predictedLabels, correctLabels, n_classes):
    """
    Compute confusion matrix given an array representing the predicted labels and an array representing correct labels
    :param predictedLabels: numpy array of shape (n,)
    :param correctLabels: numpy array of shape (n,)
    :param n_classes: number of classes for the target task for which I compute the confusion matrix
    :return: confusion matrix M with shape (n_classes, n_classes)
    """
    M = numpy.zeros((n_classes, n_classes), dtype=int)
    numpy.add.at(M, (predictedLabels, correctLabels), 1)
    return M


def compute_optimal_B_decisions(LLRs, p, C_fn, C_fp):
    P_eff = p * C_fn / (p * C_fn + (1 - p) * C_fp)
    LLRs = LLRs + numpy.log(P_eff / (1 - P_eff))
    predictedLabels = numpy.zeros_like(LLRs, dtype=int)
    predictedLabels[LLRs > 0] = 1
    return predictedLabels


def compute_un_normalized_DCF(M, p, C_fn, C_fp):
    """
    Compute un-normalized Detection Cost Function (DCF) for a binary classifier
    :param M: numpy array with shape (2, 2) representing the confusion matrix.
    :param p: prior probability of True class (class 1)
    :param C_fn: cost of false negative
    :param C_fp: cost of false positive
    :return: un-normalized DCF
    """
    FNR = M[0][1] / (M[0][1] + M[1][1])
    FPR = M[1][0] / (M[0][0] + M[1][0])
    return p * C_fn * FNR + (1 - p) * C_fp * FPR


def compute_normalized_DCF(M, p, C_fn, C_fp):
    """
    Compute normalized Detection Cost Function (DCF) for a binary classifier
    :param M: numpy array with shape (2, 2) representing the confusion matrix.
    :param p: prior probability of True class (class 1)
    :param C_fn: cost of false negative
    :param C_fp: cost of false positive
    :return: normalized DCF
    """
    FNR = M[0][1] / (M[0][1] + M[1][1])
    FPR = M[1][0] / (M[0][0] + M[1][0])
    DCF_u = p * C_fn * FNR + (1 - p) * C_fp * FPR
    return DCF_u / (min(p * C_fn, (1 - p) * C_fp))


def compute_predictions(S, t):
    """
    Compute predictions of a binary classifier given an array of scores and a threshold
    :param S: numpy array of scores with shape (n,)
    :param t: threshold used to compute the prediction
    :return: numpy array of int with shape (n,) containing the predicted labels (either 1 or 0)
    """
    predictedLabels = numpy.zeros_like(S)
    predictedLabels[S > t] = 1
    return predictedLabels.astype(int)


def compute_min_DCF(p, C_fn, C_fp, S, L):
    """
    Calculate min Detection Cost Function (DCF) for a binary classifier
    :param p:prior probability of class True (class 1)
    :param C_fn:cost of false negative
    :param C_fp:cost of false positive
    :param S:numpy array of test scores with shape (n,)
    :param L:numpy array of correct labels with shape (n,)
    :return:min DCF
    """
    sorted_S = numpy.sort(S)
    DCFs = []
    for t in sorted_S:
        predictedLabels = compute_predictions(S, t)
        M = compute_confusion_matrix(predictedLabels, L, 2)
        DCF = compute_normalized_DCF(M, p, C_fn, C_fp)
        DCFs.append(DCF)
    return min(DCFs)


def plot_DET_curve(S, L, label_names, name="default", save=False):
    """
    Plot DET curve for a binary classifier. FNR versus FPR is plotted as the threshold varies (increases)
    :param save: set to true to permanently save figure at path :param name.
    :param name: name for the saved file
    :param S: numpy array of scores with shape (n,)
    :param L: numpy array of correct labels with shape (n,)
    :param label_names: list of strings containing the names of the labels
    """
    plt.figure()
    for i in range(len(S)):
        x_DET = []
        y_DET = []
        print(label_names[i])
        print(S[i])
        for t in numpy.sort(S[i]):
            predictedLabels = compute_predictions(S[i], t)
            M = compute_confusion_matrix(predictedLabels, L, 2)
            FNR = M[0][1] / (M[0][1] + M[1][1]) * 100
            FPR = M[1][0] / (M[0][0] + M[1][0]) * 100
            x_DET.append(FPR)
            y_DET.append(FNR)
        # remove first and last element in x and y to avoid infinite values
        x_DET = x_DET[1:-1]
        y_DET = y_DET[1:-1]
        plt.plot(numpy.log(x_DET), numpy.log(y_DET), label=label_names[i])
    plt.xlabel("log(FPR)")
    plt.ylabel("log(FNR)")
    plt.legend()
    if save:
        plt.savefig(name)
    plt.show()


def plot_ROC_curve(S, L, name="default", save=False):
    """
    Plot roc curve for a binary classifier. FPR versus TPR is plotted as the threshold varies (increases)
    :param save: set to true to permanently save figure at path :param name.
    :param name: name for the saved file
    :param S: numpy array of scores with shape (n,)
    :param L: numpy array of correct labels with shape (n,)
    """
    x_ROC = []
    y_ROC = []
    for t in numpy.sort(S):
        predictedLabels = compute_predictions(S, t)
        M = compute_confusion_matrix(predictedLabels, L, 2)
        FNR = M[0][1] / (M[0][1] + M[1][1])
        FPR = M[1][0] / (M[0][0] + M[1][0])
        TPR = 1 - FNR
        x_ROC.append(FPR)
        y_ROC.append(TPR)
    plt.figure()
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(x_ROC, y_ROC)
    if save:
        plt.savefig("ROC_curves/" + name + "ROC.png")
    plt.show()


def print_bayes_error_plot(effPriorLogOdds, S, L, name, xlim=None, ylim=None, save=False):
    """
    Print the bayes error plot for a binary classifier
    :param effPriorLogOdds: list of prior log-odds (log(p/(1-p))
    :param S: numpy array of scores with shape (n,)
    :param L: numpy array of correct labels with shape (n,)
    :param name: name for the saved file
    :param save: set to true to permanently save figure at path :param name.
    :param xlim: xlim for the plot as a 2 elements list
    :param ylim: ylim for the plot as a 2 elements list
    """
    dcfs = []
    min_dcfs = []
    for p in effPriorLogOdds:
        eff_p = 1 / (1 + numpy.exp(-p))
        predictedLabels = compute_predictions(S, -p)
        M = compute_confusion_matrix(predictedLabels, L, 2)
        minDCF = compute_min_DCF(eff_p, 1, 1, S, L)
        dcf = compute_normalized_DCF(M, eff_p, 1, 1)
        dcfs.append(dcf)
        min_dcfs.append(minDCF)
        print("p = " + str(eff_p) + " DCF = " + str(dcf) + " minDCF = " + str(minDCF))

    plt.plot(effPriorLogOdds, dcfs, label="actual DCF", color="r")
    plt.plot(effPriorLogOdds, min_dcfs, label="min DCF", color="b")
    plt.xlabel("log(p/(1-p))")
    plt.legend()
    if xlim is not None:
        plt.ylim(ylim)
    if ylim is not None:
        plt.xlim(xlim)
    if save:
        plt.savefig(name)
    plt.show()


def compute_accuracy_from_CM(CM):
    """
    Compute accuracy from a confusion matrix for a binary task
    :param CM: numpy array with shape (2,2)
    :return: Accuracy value
    """
    return (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[0][1] + CM[1][0] + CM[1][1])


def print_evaluations(M, p, C_fn, C_fp, S, L):
    """
    Prints confusion matrix and min DCF for a binary task given a set of parameters
    :param M: Confusion matrix
    :param p: prior probability of class true for the chosen working point
    :param C_fn: cost of false negative for the chosen working point
    :param C_fp: cost of false positives for the chosen working point
    :param S: numpy array of scores (used to evaluate DCF_min) with shape (n_samples,)
    :param L: numpy array of labels (used to evaluate DCF_min) with shape (n_samples,
    :return:
    """
    # print(M)
    print("Min DCF:" + str(compute_min_DCF(p, C_fn, C_fp, S, L)))

# ---------------------------------------------------------------------------------------------
