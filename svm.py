import matplotlib.pyplot as plt
import numpy
import scipy
import time
from utils import *


def RBF_kernel_analysis():
    """
    Cross validation using k-fold to find the best hyperparameters for SVM with RBF kernel
    (Some values need to be changed manually to test different hyperparameters)
    """
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    C = [10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100]
    gammas = [10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-4), 10 ** (-3), 10 ** (-2)]
    PCA_i_DCFs = []
    P = calculate_pca_proj_matrix(DTR, 6)
    D_origin = numpy.dot(P.T, DTR)
    for i in range(len(gammas)):
        DTR = D_origin
        g = gammas[i]
        if i < 3:
            DTR = z_normalization(DTR)

        min_DCFs = []
        kernel_func = kernel_RBF(g)
        k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
        for c in C:
            S, L = k_fold_validation_SVM(DTR, LTR, 5, c, p_eff, split_db_k_to_1_v2, k)  # 1 is K**2
            print("minDCF:" + str(compute_min_DCF(p_eff, 1, 1, S, L)))
            min_DCFs.append(compute_min_DCF(p_eff, 1, 1, S, L))
        PCA_i_DCFs.append(min_DCFs)

    # write PCA_i_DCFs to txt file
    with open("svm/PCA_i_DCFs_RBF_PCA6_znorm.txt", "w") as f:
        for line in PCA_i_DCFs:
            f.write(str(line) + "\n")
        f.close()

    plt.figure()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    x_values = range(len(C))
    plt.xticks(x_values, C)

    # recreate PCA_iDCFS, reading from the txt file
    PCA_i_DCFs = []
    with open("svm/PCA_i_DCFs_RBF_PCA6_znorm.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("[").strip("]").strip("\n").split(",")
            line[-1] = line[-1].strip("]")
            line = [float(i) for i in line]
            PCA_i_DCFs.append(line)
        f.close()
    for i in range(len(gammas)):
        plt.plot(x_values, PCA_i_DCFs[i], label="RBF(" + str(gammas[i]) + "),Z-norm" if i < 3 else "RBF(" + str(gammas[i]) + ")")

    plt.grid()
    plt.legend()
    plt.savefig("svm/RBF_PCA6_znorm.png")
    plt.show()


def poly_kernel_analysis():
    """
    Cross validation using k-fold to finde the best hyperparameters for SVM with polynomial kernel
    (Some values need to be changed manually to test different hyperparameters)
    """
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    D_origin = DTR
    PCA_i_DCFs = []
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    C = [10 ** 2, 10, 1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    C = [10, 1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    C = C[::-1]
    kernel_func = kernel_poly(4, 1)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    for i in [0, 4]:
        min_DCFs = []
        if i == 0:
            print("No-PCA:")
            P = calculate_pca_proj_matrix(D_origin, 6)
            DTR = numpy.dot(P.T, D_origin)
        else:
            P = calculate_pca_proj_matrix(D_origin, 10 - i)
            DTR = numpy.dot(P.T, D_origin)
            print("PCA-" + str(10 - i) + ":")
            DTR = z_normalization(DTR)
        for c in C:
            S, L = k_fold_validation_SVM(DTR, LTR, 5, c, p_eff, split_db_k_to_1_v2, k)  # 1 is K**2
            print("minDCF:" + str(compute_min_DCF(p_eff, 1, 1, S, L)))
            min_DCFs.append(compute_min_DCF(p_eff, 1, 1, S, L))
        PCA_i_DCFs.append(min_DCFs)

    # write PCA_i_DCFs to txt file
    with open("svm/PCA_i_DCFs_poly_4_z_norm_comp.txt", "w") as f:
        for line in PCA_i_DCFs:
            f.write(str(line) + "\n")
        f.close()
    plt.figure()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    x_values = range(len(C))
    plt.xticks(x_values, C)

    # recreate PCA_iDCFS, reading from the txt file
    PCA_i_DCFs = []
    with open("svm/PCA_i_DCFs_poly_4_z_norm_comp.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("[").strip("]").strip("\n").split(",")
            line[-1] = line[-1].strip("]")
            line = [float(i) for i in line]
            PCA_i_DCFs.append(line)
        f.close()
    for i in range(2):
        if i == 0:
            plt.plot(x_values, PCA_i_DCFs[i], label="Poly(4)-PCA-6")
        else:
            plt.plot(x_values, PCA_i_DCFs[i], label="Poly(4)-PCA-6(Z-norm)")
    plt.grid()
    plt.legend()
    plt.savefig("svm/poly4_z_norm_comp_svm.png")
    plt.show()


def different_priors_analysis():
    """Analysys of different priors through cross-validation using training set"""
    DTR, LTR = load("Train.txt")
    # DTE, LTE = load("test.txt")
    DTR = center_Dataset(DTR)
    p_eff1 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    p_eff4 = 0.2
    p_eff5 = 0.344
    D2 = DTR
    for p in [p_eff1, p_eff2, p_eff3, p_eff4, p_eff5]:
        print("Model trained with p_eff = " + str(p))
        # PCA-6
        P = calculate_pca_proj_matrix(DTR, 6)
        D1 = numpy.dot(P.T, DTR)
        kernel_func = kernel_poly(2, 1)
        k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
        S, L = k_fold_validation_SVM(D1, LTR, 5, 0.01, p, split_db_k_to_1_v2, k)  # 1 is K**2 (best model with poly ker)
        print(str(p_eff1) + "-minDCF:" + str(compute_min_DCF(p_eff1, 1, 1, S, L)))
        print(str(p_eff2) + "-minDCF:" + str(compute_min_DCF(p_eff2, 1, 1, S, L)))
        print(str(p_eff3) + "-minDCF:" + str(compute_min_DCF(p_eff3, 1, 1, S, L)))

        # no-PCA
        kernel_func = kernel_RBF(0.001)
        k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
        S, L = k_fold_validation_SVM(D2, LTR, 5, 10, p, split_db_k_to_1_v2, k)  # 1 is K**2 (best model with poly RBF)
        print(str(p_eff1) + "-minDCF:" + str(compute_min_DCF(p_eff1, 1, 1, S, L)))
        print(str(p_eff2) + "-minDCF:" + str(compute_min_DCF(p_eff2, 1, 1, S, L)))
        print(str(p_eff3) + "-minDCF:" + str(compute_min_DCF(p_eff3, 1, 1, S, L)))


def SVM_validation_on_test_pca():
    """
    This function is used to validate the SVM model on the test set, to find what would have been the optimal pca dimension
    """
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    DTR_origin = DTR
    DTE_origin = DTE
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_t_emp = LTR[LTR == 1].shape[0] / LTR.shape[0]
    p_t = 0.167
    gammas = [10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100]
    Cs = [10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100]
    gamma_minDCFs = []
    # for i in [1, 2, 3, 4]:
    #     print("PCA = " + str(i))
    #     minDCFs = []
    #     kernel_func = kernel_RBF(0.001)
    #     k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    #     P = calculate_pca_proj_matrix(DTR_origin, 10 - i)
    #     DTR = numpy.dot(P.T, DTR_origin)
    #     DTE = numpy.dot(P.T, DTE_origin)
    #     for C in Cs:
    #         print("C = " + str(C))
    #         S = compute_SVM_scores(DTR, LTR, DTE, C * p_t / p_t_emp, C * (1 - p_t) / (1 - p_t_emp), k)
    #
    #         # calibration with best model prior=0.091, lambda = 0.0001
    #         l = 0.0001
    #         prior = 0.091
    #         S = S.reshape(1, S.shape[0])
    #         w, b = compute_w_b_with_priors(S, LTE, l, prior)
    #         S_new = numpy.dot(w.T, S) + b
    #
    #         # recover llrs from LR scores
    #         S_new = S_new - numpy.log(prior / (1 - prior))
    #
    #         minDCF = compute_min_DCF(p_eff, 1, 1, S_new, LTE)
    #         print("minDCF:" + str(minDCF))
    #         minDCFs.append(minDCF)
    #         # predictedLabels = compute_predictions(S_new, -numpy.log(p_eff / (1 - p_eff)))
    #         # M = compute_confusion_matrix(predictedLabels, LTE, 2)
    #         # print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))
    #     gamma_minDCFs.append(minDCFs)
    #
    # # write gamma_minDCFs to txt file
    # with open("svm/RBF_valid_on_test_DCFs_pca_g=0.001.txt", "w") as f:
    #     for line in gamma_minDCFs:
    #         f.write(str(line) + "\n")
    #     f.close()

    plt.figure()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    x_values = range(len(gammas))
    plt.xticks(x_values, gammas)

    # recreate PCA_iDCFS, reading from the txt file
    gamma_minDCFs = []
    with open("svm/RBF_valid_on_test_DCFs_p=0.048.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("[").strip("]").strip("\n").split(",")
            line[-1] = line[-1].strip("]")
            line = [float(i) for i in line]
            gamma_minDCFs.append(line)
        f.close()
    # search the min in gamma_minDCFs which is a list of list
    minDCFs = []
    for i in range(4):
        minDCFs.append(min(gamma_minDCFs[i]))
    best_min = min(minDCFs)
    best_min = round(best_min, 3)
    for i in range(4):
        plt.plot(x_values, gamma_minDCFs[i], label="PCA=" + str(10-i-1))

    plt.grid()
    plt.title("bestMinDCF=" + str(best_min))
    plt.legend()
    plt.savefig("svm/RBF_valid_on_test_DCFs_p=0.048.png")
    plt.show()


def SVM_validation_on_test():
    """Validation on test set, to find what would have been the optimal hyperparameters for the SVM"""
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_t_emp = LTR[LTR == 1].shape[0] / LTR.shape[0]
    p_t = 0.167
    gammas = [10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100]
    Cs = [10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100]
    gamma_minDCFs = []
    for gamma in gammas:
        print("gamma = " + str(gamma))
        minDCFs = []
        kernel_func = kernel_RBF(gamma)
        k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
        for C in Cs:
            print("C = " + str(C))
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
            minDCFs.append(minDCF)
            # predictedLabels = compute_predictions(S_new, -numpy.log(p_eff / (1 - p_eff)))
            # M = compute_confusion_matrix(predictedLabels, LTE, 2)
            # print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))
        gamma_minDCFs.append(minDCFs)

    # write gamma_minDCFs to txt file
    with open("svm/RBF_valid_on_test_DCFs_p=0.048.txt", "w") as f:
        for line in gamma_minDCFs:
            f.write(str(line) + "\n")
        f.close()

    plt.figure()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    x_values = range(len(gammas))
    plt.xticks(x_values, gammas)

    # recreate PCA_iDCFS, reading from the txt file
    gamma_minDCFs = []
    with open("svm/RBF_valid_on_test_DCFs_p=0.048.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("[").strip("]").strip("\n").split(",")
            line[-1] = line[-1].strip("]")
            line = [float(i) for i in line]
            gamma_minDCFs.append(line)
        f.close()
    # search the min in gamma_minDCFs which is a list of list
    minDCFs = []
    for i in range(len(gamma_minDCFs)):
        minDCFs.append(min(gamma_minDCFs[i]))
    best_min = min(minDCFs)
    for i in range(len(gammas)):
        plt.plot(x_values, gamma_minDCFs[i], label="gamma=" + str(gammas[i]))

    plt.grid()
    plt.title("bestMinDCF=" + str(best_min))
    plt.legend()
    plt.savefig("svm/RBF_valid_on_test_DCFs_p=0.048.png")
    plt.show()


def SVM_evaluation_on_test():
    """
    Evaluate the SVM model on the test set (best-SVM model also calibrated)
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

    return S_new


def best_model_bayes_error_plot():
    DTR, LTR = load("Train.txt")
    # DTE, LTE = load("test.txt")
    DTR = center_Dataset(DTR)
    p_t = 0.167
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    D2 = DTR
    # NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    kernel_func = kernel_RBF(0.001)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S, L = k_fold_validation_SVM(D2, LTR, 5, 10, p_t, split_db_k_to_1_v2, k)  # 1 is K**2 (best model with poly RBF)
    # effPriorLogOdds = numpy.linspace(-3, 3, 100)
    # print_bayes_error_plot(effPriorLogOdds, S, L, "calibration/RBF_ker_b_error.png", save=True)

    print("minDCF:" + str(compute_min_DCF(p_eff, 1, 1, S, L)))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff / (1 - p_eff)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))

    print(p_eff2)
    print("minDCF:" + str(compute_min_DCF(p_eff2, 1, 1, S, L)))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff2 / (1 - p_eff2)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff2, 1, 1)))

    print(p_eff3)
    print("minDCF:" + str(compute_min_DCF(p_eff3, 1, 1, S, L)))
    predictedLabels = compute_predictions(S, -numpy.log(p_eff3 / (1 - p_eff3)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff3, 1, 1)))


def calibrated_best_model_bayes_error_plot():
    """
    Bayes error plot of the calibrated version of the best svm-model, used for the report
    """
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_t = 0.167
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    D2 = DTR
    # NO-PCA, gamma = 0.001, C = 10, p_t = 0.167
    kernel_func = kernel_RBF(0.001)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S, L = k_fold_validation_SVM(D2, LTR, 5, 10, p_t, split_db_k_to_1_v2, k)  # 1 is K**2 (best model with poly RBF)

    # calibration with best model prior=0.091, lambda = 0.0001
    l = 0.0001
    prior = 0.091
    S = S.reshape(1, S.shape[0])
    w, b = compute_w_b_with_priors(S, L, l, prior)
    S_new = numpy.dot(w.T, S) + b

    # recover llrs from LR scores
    S_new = S_new - numpy.log(prior / (1 - prior))

    minDCF = compute_min_DCF(p_eff, 1, 1, S_new, L)
    print("minDCF:" + str(minDCF))
    predictedLabels = compute_predictions(S_new, -numpy.log(p_eff / (1 - p_eff)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff, 1, 1)))

    print(p_eff2)
    print("minDCF:" + str(compute_min_DCF(p_eff2, 1, 1, S_new, L)))
    predictedLabels = compute_predictions(S_new, -numpy.log(p_eff2 / (1 - p_eff2)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff2, 1, 1)))

    print(p_eff3)
    print("minDCF:" + str(compute_min_DCF(p_eff3, 1, 1, S_new, L)))
    predictedLabels = compute_predictions(S_new, -numpy.log(p_eff3 / (1 - p_eff3)))
    M = compute_confusion_matrix(predictedLabels, L, 2)
    print("actualDCF:" + str(compute_normalized_DCF(M, p_eff3, 1, 1)))

    effPriorLogOdds = numpy.linspace(-3, 3, 100)
    print_bayes_error_plot(effPriorLogOdds, S_new, L, "calibration/RBF_ker_calibrated_b_error.png", save=True)


def SVM_best_model_calibration():
    """
    Performs cross validation to find the best calibration model for our best SVM model, used for the report
    """
    DTR, LTR = load("Train.txt")
    # DTE, LTE = load("test.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    p_eff4 = 0.2
    p_eff5 = 0.344
    D2 = DTR
    # NO-PCA, gamma = 0.001, C = 10, p_t= 0.167
    p_t = 0.167
    kernel_func = kernel_RBF(0.001)
    k = lambda D1, D2: kernel_func(D1, D2) + 1 ** 2
    S, L = k_fold_validation_SVM(D2, LTR, 5, 10, p_t, split_db_k_to_1_v2, k)  # 1 is K**2 (best model with poly RBF) # now i have my scores
    # we train the calibration model, and we use k_fold approach to evaluate it (S, L are reshuffled inside the splid_db_k_to_1_v2 function)
    delta = 10000
    hiperParams = (1.0, 1.0)
    S = S.reshape(1, S.shape[0])
    l_prior_deltas = []
    lambdas = [10 ** (-4)]
    priors = [p_eff, p_eff2, p_eff3, p_eff4, p_eff5]
    for l in lambdas:
        print("lambda:", l)
        act_dcfs = []
        for p_t in priors:
            print(p_t)
            S_new, L_new = k_fold_validation_logreg(S, L, 5, l, p_t, split_db_k_to_1_v2)
            # effPriorLogOdds = numpy.linspace(-3, 2, 100)
            # print_bayes_error_plot(effPriorLogOdds, S_new, L_new, "calibration/RBF_ker_calibrated_b_error.png", save=True)
            minDCF = compute_min_DCF(p_eff, 1, 1, S_new, L_new)
            print("minDCF:" + str(minDCF))

            # recover llrs from LR scores
            S_new = S_new - numpy.log(p_t / (1 - p_t))

            predictedLabels = compute_predictions(S_new, -numpy.log(p_eff / (1 - p_eff)))
            M = compute_confusion_matrix(predictedLabels, L_new, 2)
            actDCF = compute_normalized_DCF(M, p_eff, 1, 1)
            print("actualDCF:" + str(actDCF))
            if numpy.abs(actDCF - minDCF) < delta:
                delta = numpy.abs(actDCF - minDCF)
                hiperParams = (l, p_t)
            act_dcfs.append(actDCF)
        l_prior_deltas.append(act_dcfs)
    print(l_prior_deltas)

    # plot l_prior_deltas with p_t on the x, and different curves for different l

    plt.figure()
    plt.xlabel("embedded prior")
    plt.ylabel("actualDCF")
    priors = [round(p, 3) for p in priors]
    x_values = range(len(priors))
    plt.xticks(x_values, priors)
    for i in range(len(l_prior_deltas)):
        plt.plot(x_values, l_prior_deltas[i], label="lambda = " + str(lambdas[i]))

    plt.grid()
    plt.legend()
    plt.savefig("calibration/calib_cross_val_zoom.png")
    plt.show()

    print("best lambda:", hiperParams[0])
    print("best p_t:", hiperParams[1])
    print("delta:", delta)


if __name__ == '__main__':
    start_time = time.perf_counter()
    # SVM_best_model_calibration()
    # best_model_bayes_error_plot()
    # calibrated_best_model_bayes_error_plot()
    # SVM_evaluation_on_test()
    # SVM_validation_on_test()
    SVM_validation_on_test_pca()
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print("Execution time:", execution_time, "seconds")

    pass
