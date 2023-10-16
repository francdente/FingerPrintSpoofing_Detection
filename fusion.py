import matplotlib.pyplot as plt
import numpy
import scipy
import time
from utils import *


# Different fusion models
def GMM_SVM():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)

    # GMM 2-TD, 8-TD, PCA-6
    S_GMM, L_GMM = best_GMM_score()

    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    S_SVM, L_SVM = best_SVM_score()

    S_fusion = numpy.vstack([S_SVM, S_GMM])
    L_fusion = L_SVM
    lambdas = [1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    prior = 0.091  # as seen in calibration
    minDCFs = []
    actDCFs = []
    for l in lambdas:
        S, L, minDCF, actDCF = fusion_LR(S_fusion, L_fusion, l, prior)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        if l == 0.01:
            effPriorLogOdds = numpy.linspace(-3, 3, 100)
            # print_bayes_error_plot(effPriorLogOdds, S, L, "fusion/GMM_SVM_b_error.png", save=True)

    plt.figure()
    plt.ylabel("DCF")
    plt.xlabel("lambda")
    x_values = range(len(lambdas))
    plt.xticks(x_values, lambdas)

    plt.plot(x_values, minDCFs, label="minDCF")
    plt.plot(x_values, actDCFs, label="actDCF")

    plt.grid()
    plt.legend()
    plt.savefig("fusion/GMM_SVM.png")
    plt.show()

    pass


def Q_LR_and_GMM():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)

    # Q-LogReg p_t = 0.091, lambda = 0.1, PCA-6
    S_LR, L_LR = best_Q_LR_score()

    # GMM 2-TD, 8-TD, PCA-6
    S_GMM, L_GMM = best_GMM_score()

    S_fusion = numpy.vstack([S_GMM, S_LR])
    L_fusion = L_GMM
    lambdas = [1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    prior = 0.091  # as seen in calibration
    minDCFs = []
    actDCFs = []
    for l in lambdas:
        S, L, minDCF, actDCF = fusion_LR(S_fusion, L_fusion, l, prior)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        if l == 0.01:
            effPriorLogOdds = numpy.linspace(-3, 3, 100)
            # print_bayes_error_plot(effPriorLogOdds, S, L, "fusion/QLR_GMM_b_error.png", save=True)

    plt.figure()
    plt.ylabel("DCF")
    plt.xlabel("lambda")
    x_values = range(len(lambdas))
    plt.xticks(x_values, lambdas)

    plt.plot(x_values, minDCFs, label="minDCF")
    plt.plot(x_values, actDCFs, label="actDCF")

    plt.grid()
    plt.legend()
    plt.savefig("fusion/QLR_GMM.png")
    plt.show()


def Q_LR_and_SVM():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)

    # Q-LogReg p_t = 0.091, lambda = 0.1, PCA-6
    S_LR, L_LR = best_Q_LR_score()

    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    S_SVM, L_SVM = best_SVM_score()

    S_fusion = numpy.vstack([S_SVM, S_LR])
    L_fusion = L_SVM
    lambdas = [1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    prior = 0.091  # as seen in calibration
    minDCFs = []
    actDCFs = []
    for l in lambdas:
        S, L, minDCF, actDCF = fusion_LR(S_fusion, L_fusion, l, prior)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        if l == 0.001:
            effPriorLogOdds = numpy.linspace(-3, 3, 100)
            # print_bayes_error_plot(effPriorLogOdds, S, L, "fusion/QLR_SVM_b_error.png", save=True)

    plt.figure()
    plt.ylabel("DCF")
    plt.xlabel("lambda")
    x_values = range(len(lambdas))
    plt.xticks(x_values, lambdas)

    plt.plot(x_values, minDCFs, label="minDCF")
    plt.plot(x_values, actDCFs, label="actDCF")

    plt.grid()
    plt.legend()
    plt.savefig("fusion/QLR_SVM.png")
    plt.show()


def Q_LR_GMM_SVM():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    S_SVM, L_SVM = best_SVM_score()

    # Q-LogReg p_t = 0.091, lambda = 0.1, PCA-6
    S_LR, L_LR = best_Q_LR_score()

    # GMM 2-TD, 8-TD, PCA-6
    S_GMM, L_GMM = best_GMM_score()

    S_fusion = numpy.vstack([S_SVM, S_LR, S_GMM])
    L_fusion = L_SVM
    lambdas = [1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    prior = 0.091  # as seen in calibration
    minDCFs = []
    actDCFs = []

    for l in lambdas:
        S, L, minDCF, actDCF = fusion_LR(S_fusion, L_fusion, l, prior)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)
        if l == 0.01:
            effPriorLogOdds = numpy.linspace(-3, 3, 100)
            # print_bayes_error_plot(effPriorLogOdds, S, L, "fusion/QLR_GMM_SVM_b_error.png", save=True)

    plt.figure()
    plt.ylabel("DCF")
    plt.xlabel("lambda")
    x_values = range(len(lambdas))
    plt.xticks(x_values, lambdas)

    plt.plot(x_values, minDCFs, label="minDCF")
    plt.plot(x_values, actDCFs, label="actDCF")

    plt.grid()
    plt.legend()
    plt.savefig("fusion/QLR_GMM_SVM.png")
    plt.show()


# ---------------------------------


def best_SVM_score_on_dataset(DTR, LTR, DTE):
    """
    Compute scores of the model trained on DTR, and scoring on DTE
    """
    DTR = center_Dataset(DTR)
    p_t = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_t_emp = LTR[LTR == 1].shape[0] / LTR.shape[0]
    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    DTR_SVM = DTR
    p_t_SVM = 0.167
    gamma = 0.001
    C = 10
    kernel_func = kernel_RBF(gamma)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S = compute_SVM_scores(DTR, LTR, DTE, C * p_t / p_t_emp, C * (1 - p_t) / (1 - p_t_emp), k)
    return S


def best_GMM_score_on_dataset(DTR, LTR, DTE):
    """
    Compute scores of the model trained on DTR, and scoring on DTE
    """
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    G_auth = 2
    G_spoof = 8
    diagonal = [True, True]
    tied_covariance = [True, True]
    S = generate_GMM_scores(DTR, LTR, DTE, [G_spoof, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
    return S


# scoring on validation set (k-fold cross validation) of our 3 chosen models
def best_SVM_score():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    DTR_SVM = DTR
    p_t_SVM = 0.167
    gamma = 0.001
    C = 10
    kernel_func = kernel_RBF(gamma)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S_SVM, L_SVM = k_fold_validation_SVM(DTR_SVM, LTR, 5, C, p_t_SVM, split_db_k_to_1_v2, k)
    return S_SVM, L_SVM


def best_Q_LR_score():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    # Q-LogReg p_t = 0.091, lambda = 0.1, PCA-6
    l = 0.1
    p_t_LR = 0.091
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR_LR = numpy.dot(P.T, DTR)
    DTR_LR = to_expanded_feature_space(DTR_LR)
    DTR_LR = center_Dataset(DTR_LR)
    S_LR, L_LR = k_fold_validation_logreg(DTR_LR, LTR, 5, l, p_t_LR, split_db_k_to_1_v2)
    # recover llrs from LR scores
    S_LR = S_LR - numpy.log(p_t_LR / (1 - p_t_LR))
    return S_LR, L_LR


def best_GMM_score():
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    # GMM 2-TD, 8-TD, PCA-6
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR_GMM = numpy.dot(P.T, DTR)
    G_auth = 2
    G_spoof = 8
    diagonal = [True, True]
    tied_covariance = [True, True]
    S_GMM, L_GMM = k_fold_validation_GMM(DTR_GMM, LTR, 5, split_db_k_to_1_v2, [G_spoof, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
    return S_GMM, L_GMM


# -------------------------------------------

def fusion_LR(S_fusion, L_fusion, l, prior):
    """
    Fusion of the models using linear logistic regression starting from the scores of the models
    """
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)

    S, L = k_fold_validation_logreg(S_fusion, L_fusion, 5, l, prior, split_db_k_to_1_v2)
    print("lambda = ", l, "p_t = ", prior)
    minDCF = compute_min_DCF(p_eff, 1, 1, S, L)
    print("minDCF = ", minDCF)
    minDCF = compute_min_DCF(p_eff2, 1, 1, S, L)
    print("minDCF2 = ", minDCF)
    minDCF = compute_min_DCF(p_eff3, 1, 1, S, L)
    print("minDCF3 = ", minDCF)

    # recover llrs from LR scores
    S = S - numpy.log(prior / (1 - prior))

    predictedLabels = compute_predictions(S, -numpy.log(p_eff / (1 - p_eff)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    actDCF = compute_normalized_DCF(M, p_eff, 1, 1)
    print("actDCF = ", actDCF)

    predictedLabels = compute_predictions(S, -numpy.log(p_eff2 / (1 - p_eff2)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    actDCF = compute_normalized_DCF(M, p_eff2, 1, 1)
    print("actDCF2 = ", actDCF)

    predictedLabels = compute_predictions(S, -numpy.log(p_eff3 / (1 - p_eff3)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    actDCF = compute_normalized_DCF(M, p_eff3, 1, 1)
    print("actDCF3 = ", actDCF)

    return S, L, minDCF, actDCF


# function regarding the best fusion model we have chosen (GMM + SVM)

def best_GMM_score_postEval_on_dataset(DTR, LTR, DTE):
    """
    Compute scores of the model trained on DTR, and scoring on DTE, it uses optimal params estimated in post eval analysis
    """
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    G_auth = 2
    G_spoof = 16
    diagonal = [True, True]
    tied_covariance = [True, False]
    S = generate_GMM_scores(DTR, LTR, DTE, [G_spoof, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
    return S


def best_SVM_score_postEval_on_dataset(DTR, LTR, DTE):
    """
    Compute scores of the model trained on DTR, and scoring on DTE, it uses optimal params estimated in post eval analysis
    """
    # DTR = center_Dataset(DTR)
    p_t = 0.167
    p_t_emp = LTR[LTR == 1].shape[0] / LTR.shape[0]
    # RBF-SVM PCA-6, gamma = 0.01, C = 0.001, p_t = 0.167
    gamma = 0.01
    C = 0.001
    kernel_func = kernel_RBF(gamma)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S = compute_SVM_scores(DTR, LTR, DTE, C * p_t / p_t_emp, C * (1 - p_t) / (1 - p_t_emp), k)
    return S


def GMM_SVM_evaluation_postEval_on_test():
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)

    # GMM 2-D, 16-TD, NO-PCA
    DTR_GMM = DTR
    STR_GMM = best_GMM_score_postEval_on_dataset(DTR_GMM, LTR, DTR_GMM)

    # RBF-SVM PCA-6, gamma = 0.01, C = 0.001, p_t = 0.167
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR_SVM = numpy.dot(P.T, DTR)
    STR_SVM = best_SVM_score_postEval_on_dataset(DTR_SVM, LTR, DTR_SVM)

    DTE_GMM = DTE
    DTE_SVM = numpy.dot(P.T, DTE)
    STE_GMM = best_GMM_score_postEval_on_dataset(DTR_GMM, LTR, DTE_GMM)
    STE_SVM = best_SVM_score_postEval_on_dataset(DTR_SVM, LTR, DTE_SVM)

    S_fusion = numpy.vstack([STR_SVM, STR_GMM])
    STE_fusion = numpy.vstack([STE_SVM, STE_GMM])
    L_fusion = LTR
    l = 0.001
    prior = 0.091  # as seen in calibration

    S = compute_logreg_scores_with_priors(S_fusion, L_fusion, STE_fusion, l, prior)
    # recover llrs from LR scores
    S = S - numpy.log(prior / (1 - prior))

    minDCF = compute_min_DCF(p_eff, 1, 1, S, LTE)
    print("minDCF:" + str(minDCF))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff / (1 - p_eff)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))

    return S


def GMM_SVM_evaluation_on_test():
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)

    # GMM 2-TD, 8-TD, PCA-6
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR_GMM = numpy.dot(P.T, DTR)
    STR_GMM = best_GMM_score_on_dataset(DTR_GMM, LTR, DTR_GMM)

    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    DTR_SVM = DTR
    STR_SVM = best_SVM_score_on_dataset(DTR_SVM, LTR, DTR_SVM)

    DTE_GMM = numpy.dot(P.T, DTE)
    DTE_SVM = DTE
    STE_GMM = best_GMM_score_on_dataset(DTR_GMM, LTR, DTE_GMM)
    STE_SVM = best_SVM_score_on_dataset(DTR_SVM, LTR, DTE_SVM)

    S_fusion = numpy.vstack([STR_SVM, STR_GMM])
    STE_fusion = numpy.vstack([STE_SVM, STE_GMM])
    L_fusion = LTR
    l = 0.01
    prior = 0.091  # as seen in calibration

    S = compute_logreg_scores_with_priors(S_fusion, L_fusion, STE_fusion, l, prior)
    # recover llrs from LR scores
    S = S - numpy.log(prior / (1 - prior))

    minDCF = compute_min_DCF(p_eff, 1, 1, S, LTE)
    print("minDCF:" + str(minDCF))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff / (1 - p_eff)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))

    minDCF2 = compute_min_DCF(p_eff2, 1, 1, S, LTE)
    print("minDCF2:" + str(minDCF2))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff2 / (1 - p_eff2)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff2, 1, 1)))

    minDCF3 = compute_min_DCF(p_eff3, 1, 1, S, LTE)
    print("minDCF3:" + str(minDCF3))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff3 / (1 - p_eff3)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff3, 1, 1)))

    return S


if __name__ == '__main__':
    GMM_SVM_evaluation_postEval_on_test()
    # print("GMM+SVM:")
    # GMM_SVM()
    # print("Q-LR+GMM:")
    # Q_LR_and_GMM()
    # print("Q-LR+SVM:")
    # Q_LR_and_SVM()
    # print("Q-LR+GMM+SVM:")
    # Q_LR_GMM_SVM()
    # Q_LR_GMM_calibration()
