import matplotlib.pyplot as plt
import numpy
import scipy
import time
from utils import *
from fusion import *
from gmm import *
from svm import *
from logistic_regression import *


def GMM_SVM_evaluation_on_test():
    """Evaluate our GMM_SVM chosen model on the test set, after training on the training set"""
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

    return S, LTE


def SVM_evaluation_on_test():
    """
    Evaluate the SVM model on the test set (best-SVM model also calibrated) after training on the training set
    """
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    p_t_emp = LTR[LTR == 1].shape[0] / LTR.shape[0]
    # NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    p_t = 0.167
    C = 10
    kernel_func = kernel_RBF(0.001)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S = compute_SVM_scores(DTR, LTR, DTE, C * p_t / p_t_emp, C * (1 - p_t) / (1 - p_t_emp), k)

    # calibration with best model prior=0.091, lambda = 0.0001
    l = 0.0001
    prior = 0.091
    S = S.reshape(1, S.shape[0])
    w, b = compute_w_b_with_priors(S, LTE, l, prior)
    S_new = numpy.dot(w.T, S) + b

    # recover llrs from LR scores
    S_new = S_new - numpy.log(prior / (1 - prior))

    minDCF = compute_min_DCF(p_eff, 1, 1, S_new, LTE)
    print("minDCF:" + str(minDCF))
    predictedLabels = compute_predictions(S_new, -numpy.log(p_eff / (1 - p_eff)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))

    print(p_eff2)
    print("minDCF:" + str(compute_min_DCF(p_eff2, 1, 1, S_new, LTE)))
    predictedLabels = compute_predictions(S_new, -numpy.log(p_eff2 / (1 - p_eff2)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff2, 1, 1)))

    print(p_eff3)
    print("minDCF:" + str(compute_min_DCF(p_eff3, 1, 1, S_new, LTE)))
    predictedLabels = compute_predictions(S_new, -numpy.log(p_eff3 / (1 - p_eff3)))
    M = compute_confusion_matrix(predictedLabels, LTE, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff3, 1, 1)))

    return S_new, LTE


def logreg_evaluation_on_test():
    """Evaluate our QLogReg chosen model on the test set, after training on the training set"""
    # best model: Q-logreg lambda=0.1 PCA-6
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR = numpy.dot(P.T, DTR)
    DTE = numpy.dot(P.T, DTE)
    DTR = to_expanded_feature_space(DTR)
    DTE = to_expanded_feature_space(DTE)
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)

    p_t = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    S = compute_logreg_scores_with_priors(DTR, LTR, DTE, 0.1, p_t)
    # recover llrs from scores
    S = S - numpy.log(p_t / (1 - p_t))

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

    return S, LTE


def GMM_evaluation_on_test():
    """Evaluate our GMM chosen model on the test set, after training on the training set"""
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    # set up config for analysis
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR = numpy.dot(P.T, DTR)
    DTE = numpy.dot(P.T, DTE)
    diagonal = [True, True]
    tied_covariance = [True, True]
    G_auth = 2
    G_spoof = 8
    # ----------------------------

    S = generate_GMM_scores(DTR, LTR, DTE, [G_spoof, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)

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

    return S, LTE


def DET_plot_for_evaluation():
    """ DET plot of our best performing models over the evaluation/test set"""
    S_GMM_SVM, LTE = GMM_SVM_evaluation_on_test()
    S_LR, _ = logreg_evaluation_on_test()
    S_GMM, _ = GMM_evaluation_on_test()
    S_SVM, LTE = SVM_evaluation_on_test()
    print(S_GMM_SVM.shape)
    print(S_LR.shape)
    print(S_GMM.shape)
    print(S_SVM.shape)

    plot_DET_curve([S_GMM_SVM, S_LR, S_GMM, S_SVM], LTE, ["GMM-SVM", "LR", "GMM", "SVM"], "evaluation/DET_plot_for_evaluation.png", save=True)


def DET_plot_for_training():
    """ DET plot of our best performing models over the validation set taken from the training set"""
    # GMM 2-TD, 8-TD, PCA-6
    S_GMM, L_GMM = best_GMM_score()

    # RBF-SVM NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    S_SVM, L_SVM = best_SVM_score()

    S_fusion = numpy.vstack([S_SVM, S_GMM])
    L_fusion = L_SVM
    l = 0.01
    prior = 0.091  # as seen in calibration
    S_GMM_SVM, L, minDCF, actDCF = fusion_LR(S_fusion, L_fusion, l, prior)

    S_LR, _ = best_Q_LR_score()
    S_GMM, _ = best_GMM_score()
    S_SVM, _ = best_SVM_score()

    plot_DET_curve([S_GMM_SVM, S_LR, S_GMM, S_SVM], L, ["GMM-SVM", "LR", "GMM", "SVM"], "evaluation/DET_plot_for_training.png", save=True)


def eval_bayes_error_plot():
    """Print the bayes error plot of all our best performing models"""
    S_GMM_SVM, LTE = GMM_SVM_evaluation_on_test()
    S_LR, _ = logreg_evaluation_on_test()
    S_GMM, _ = GMM_evaluation_on_test()
    S_SVM, _ = SVM_evaluation_on_test()

    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    dcfs_GMM_SVM = []
    dcfs_LR = []
    dcfs_GMM = []
    dcfs_SVM = []
    min_dcfs_GMM_SVM = []
    min_dcfs_LR = []
    min_dcfs_GMM = []
    min_dcfs_SVM = []
    for p in effPriorLogOdds:
        eff_p = 1 / (1 + numpy.exp(-p))
        predictedLabels_GMM_SVM = compute_predictions(S_GMM_SVM, -p)
        predictedLabels_LR = compute_predictions(S_LR, -p)
        predictedLabels_GMM = compute_predictions(S_GMM, -p)
        predictedLabels_SVM = compute_predictions(S_SVM, -p)
        M_GMM_SVM = compute_confusion_matrix(predictedLabels_GMM_SVM, LTE, 2)
        M_LR = compute_confusion_matrix(predictedLabels_LR, LTE, 2)
        M_GMM = compute_confusion_matrix(predictedLabels_GMM, LTE, 2)
        M_SVM = compute_confusion_matrix(predictedLabels_SVM, LTE, 2)
        dcfs_GMM_SVM.append(compute_normalized_DCF(M_GMM_SVM, eff_p, 1, 1))
        dcfs_LR.append(compute_normalized_DCF(M_LR, eff_p, 1, 1))
        dcfs_GMM.append(compute_normalized_DCF(M_GMM, eff_p, 1, 1))
        dcfs_SVM.append(compute_normalized_DCF(M_SVM, eff_p, 1, 1))
        min_dcfs_GMM_SVM.append(compute_min_DCF(eff_p, 1, 1, S_GMM_SVM, LTE))
        min_dcfs_LR.append(compute_min_DCF(eff_p, 1, 1, S_LR, LTE))
        min_dcfs_GMM.append(compute_min_DCF(eff_p, 1, 1, S_GMM, LTE))
        min_dcfs_SVM.append(compute_min_DCF(eff_p, 1, 1, S_SVM, LTE))

    plt.figure()
    # do a scattered plot for minDCf
    plt.plot(effPriorLogOdds, dcfs_GMM_SVM, label="GMM_SVM(act)", color="b")
    plt.plot(effPriorLogOdds, min_dcfs_GMM_SVM, label="GMM_SVM(min)", linestyle="dashed", color="b")
    plt.plot(effPriorLogOdds, dcfs_LR, label="LR(act)", color="y")
    plt.plot(effPriorLogOdds, min_dcfs_LR, label="LR(min)", color="y", linestyle="dashed")
    plt.plot(effPriorLogOdds, dcfs_GMM, label="GMM(act)", color="g")
    plt.plot(effPriorLogOdds, min_dcfs_GMM, label="GMM(min)", color="g", linestyle="dashed")
    plt.plot(effPriorLogOdds, dcfs_SVM, label="SVM(act)", color="r")
    plt.plot(effPriorLogOdds, min_dcfs_SVM, label="SVM(min)", color="r", linestyle="dashed")
    plt.legend()
    plt.xlabel("log(p/(1-p))")
    plt.ylabel("DCF")
    plt.savefig("evaluation/eval_b_error.png")
    plt.show()


if __name__ == '__main__':
    # DET_plot_for_training()
    # DET_plot_for_evaluation()
    # eval_bayes_error_plot()
    logreg_evaluation_on_test()
