import math

import numpy
import scipy
import sklearn.datasets
from utils import *


def compute_accuracies_on_test(DTR, LTR, DTE, LTE, P):
    # Multivariate Gaussian Model
    SJoint_log = generate_joint_log_density(generate_log_score_matrix(DTR, LTR, DTE, LTE), P)
    acc, err = compute_accuracy(SJoint_log, LTE)
    err_tot_MVG = err

    # Naive Bayes Gaussian Model
    SJoint_log = generate_joint_log_density(generate_log_score_matrix(DTR, LTR, DTE, LTE, naive_bayes=True), P)
    acc, err = compute_accuracy(SJoint_log, LTE)
    err_tot_NB = err

    # Tied Covariance Model
    SJoint_log = generate_joint_log_density(generate_log_score_matrix(DTR, LTR, DTE, LTE, tied_covariance=True), P)
    acc, err = compute_accuracy(SJoint_log, LTE)
    err_tot_tied_MVG = err

    # Tied Naive Bayes
    SJoint_log = generate_joint_log_density(
        generate_log_score_matrix(DTR, LTR, DTE, LTE, naive_bayes=True, tied_covariance=True), P)
    acc, err = compute_accuracy(SJoint_log, LTE)
    err_tot_tied_NB = err

    print("Multivariate Gaussian model:")
    print(err_tot_MVG)
    print("Naive Bayes model:")
    print(err_tot_NB)
    print("Tied Covariance model:")
    print(err_tot_tied_MVG)
    print("Tied Naive Bayes:")
    print(err_tot_tied_NB)



#########################################################################################
if __name__ == '__main__':
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)

    P = calculate_pca_proj_matrix()

    # P = calculate_pca_proj_matrix(DTR, 9)
    # DTR = numpy.dot(P.T, DTR)
    # DTE = numpy.dot(P.T, DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    t = -numpy.log(p_eff / (1 - p_eff))

    D_origin = DTR
    for i in range(8):
        if i == 0:
            print("No-PCA:")
        else:
            P = calculate_pca_proj_matrix(D_origin, 10 - i)
            DTR = numpy.dot(P.T, D_origin)
            print("PCA-" + str(10 - i) + ":")

        S_MVG, S_NB, S_tied_MVG, S_tied_NB, L = k_fold_validation_gaussian(DTR, LTR, 5, split_db_k_to_1_v2)

        CM_MVG = compute_confusion_matrix(compute_predictions(S_MVG, t), L.astype(int), 2)
        CM_NB = compute_confusion_matrix(compute_predictions(S_NB, t), L.astype(int), 2)
        CM_tied_MVG = compute_confusion_matrix(compute_predictions(S_tied_MVG, t), L.astype(int), 2)
        CM_tied_NB = compute_confusion_matrix(compute_predictions(S_tied_NB, t), L.astype(int), 2)

        eff_prior_log_odds = numpy.linspace(-6, 6, 60)  # Used for the bayes error plot
        xlim = [-6, 6]
        ylim = [0, 1.1]
        print("Gaussian:")
        print_evaluations(CM_MVG, 0.5, 1, 10, S_MVG, L)
        # plot_ROC_curve(S_MVG, L, "mvg", save=True)
        # print_bayes_error_plot(eff_prior_log_odds, S_MVG, L, "mvg", xlim, ylim, save=True)

        print("Naive-Bayes:")
        print_evaluations(CM_NB, 0.5, 1, 10, S_NB, L)
        # plot_ROC_curve(S_NB, L, "nb", save=True)
        # print_bayes_error_plot(eff_prior_log_odds, S_NB, L, "nb", xlim, ylim, save=True)

        print("Tied Gaussian:")
        print_evaluations(CM_tied_MVG, 0.5, 1, 10, S_tied_MVG, L)
        # plot_ROC_curve(S_tied_MVG, L, "tied_mvg", save=True)
        # print_bayes_error_plot(eff_prior_log_odds, S_tied_MVG, L, "tied_mvg", xlim, ylim, save=True)

        print("Tied Naive-Bayes:")
        print_evaluations(CM_tied_NB, 0.5, 1, 10, S_tied_NB, L)
        # plot_ROC_curve(S_tied_NB, L, "tied_nb", save=True)
        # print_bayes_error_plot(eff_prior_log_odds, S_tied_NB, L, "tied_nb", xlim, ylim, save=True)
        print("-------------------------------------------")
