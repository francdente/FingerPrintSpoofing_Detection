import matplotlib.pyplot as plt
import numpy
import scipy
import time
from utils import *
import itertools


def GMM_analysis(diagonal=[False, False], tied_covariance=[False, False]):
    """
    During analysis, used to select the best GMM model, using k-fold cross validation on training data
    """
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)

    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)

    D_origin = DTR

    for G in [2, 4, 8, 16, 32]:
        print("G:", G)
        for i in range(6):
            if i == 0:
                DTR = D_origin
                print("No-PCA:")
            else:
                P = calculate_pca_proj_matrix(D_origin, 10 - i)
                DTR = numpy.dot(P.T, D_origin)
                print("PCA-" + str(10 - i) + ":")
            S, L = k_fold_validation_GMM(DTR, LTR, 5, split_db_k_to_1_v2, [G, 2], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
            print(p_eff, "minDCF:" + str(compute_min_DCF(p_eff, 1, 1, S, L)))
            print(p_eff2, "minDCF:" + str(compute_min_DCF(p_eff2, 1, 1, S, L)))
            print(p_eff3, "minDCF:" + str(compute_min_DCF(p_eff3, 1, 1, S, L)))


def GMM_analysis_with_plots():
    """
    During analysis, used to select the best GMM model, using k-fold cross validation on training data.
    It also generates the needed plots
    """
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    D_origin = DTR
    # set up config for analysis
    diagonal = [True, False]
    tied_covariance = [True, True]
    path_name = "gmm/gmm_G,2_TD_T"
    PCAs = [0, 1, 2, 3, 4, 5]
    Gs = [2, 4, 8, 16]
    G_auth = 2
    # ----------------------------
    tot_minDCFs = []
    PCA_minDCFs = []
    for i in PCAs:
        minDCFs = []
        if i == 0:
            DTR = D_origin
            print("No-PCA:")
        else:
            P = calculate_pca_proj_matrix(D_origin, 10 - i)
            DTR = numpy.dot(P.T, D_origin)
            print("PCA-" + str(10 - i) + ":")
        for G in Gs:
            print("G:", G)
            S, L = k_fold_validation_GMM(DTR, LTR, 5, split_db_k_to_1_v2, [G, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
            minDCF = compute_min_DCF(p_eff, 1, 1, S, L)
            print("minDCF:" + str(minDCF))
            minDCFs.append(minDCF)
            tot_minDCFs.append(minDCF)
        PCA_minDCFs.append(minDCFs)

    # write PCA_minDCFs to txt file
    with open(path_name + ".txt", "w") as f:
        for line in PCA_minDCFs:
            f.write(str(line) + "\n")
        f.write("best minDCF: " + str(min(tot_minDCFs)) + "\n")
        f.close()

    plt.figure()
    plt.ylabel("minDCF")
    plt.xlabel("G (for spoofed class)")
    x_values = range(len(Gs))
    plt.xticks(x_values, Gs)

    # recreate PCA_minDCFS, reading from the txt file
    PCA_minDCFs = []
    with open(path_name + ".txt", "r") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            line = line.strip("[").strip("]").strip("\n").split(",")
            line[-1] = line[-1].strip("]")
            line = [float(i) for i in line]
            PCA_minDCFs.append(line)
        best_minDCF = float(lines[-1].split(":")[-1].strip("\n"))
        best_minDCF = round(best_minDCF, 3)
        f.close()
    plt.title("authentic G=" + str(G_auth) + " (best minDCF: " + str(best_minDCF) + ")")
    for i in range(len(PCA_minDCFs)):
        plt.plot(x_values, PCA_minDCFs[i], label="PCA-" + str(10 - i) if i != 0 else "No-PCA")

    plt.legend()
    plt.grid()
    plt.savefig(path_name + ".png")
    plt.show()


def best_model_bayes_error_plot():
    """Bayes error plot of our chosen best GMM model"""
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    G_auth = 2
    G_spoof = 8
    diagonal = [True, True]
    tied_covariance = [True, True]
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR = numpy.dot(P.T, DTR)
    S, L = k_fold_validation_GMM(DTR, LTR, 5, split_db_k_to_1_v2, [G_spoof, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
    # effPriorLogOdds = numpy.linspace(-3, 3, 100)
    # print_bayes_error_plot(effPriorLogOdds, S, L, "calibration/GMM_b_error.png", save=True)

    print(p_eff)
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


def GMM_validation_on_test():
    """Used to generate the table in the post-evaluation analysis of GMM, to find the optimal configuration for the evaluation set"""
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    DTR_origin = DTR
    DTE_origin = DTE
    # set up config for analysis
    diagonal = [True, True]
    tied_covariance = [True, True]
    path_name = "gmm/gmm_G,2_TD_T"
    PCAs = [0, 1, 2, 3, 4]
    Gs = [4, 8, 16]
    G_auth = 2
    # ----------------------------
    tot_minDCFs = []
    PCA_minDCFs = []
    bool_list = [True, False]

    combinations = list(itertools.product(bool_list, repeat=4))
    best_min = 1000
    best_params = ()
    for combo in combinations:
        print(combo)
    for combo in combinations:
        diagonal = [combo[0], combo[1]]
        tied_covariance = [combo[2], combo[3]]
        for G_auth in [1, 2]:
            for i in PCAs:
                minDCFs = []
                if i == 0:
                    DTR = DTR_origin
                    DTE = DTE_origin
                    print("No-PCA:")
                else:
                    P = calculate_pca_proj_matrix(DTR_origin, 10 - i)
                    DTR = numpy.dot(P.T, DTR_origin)
                    DTE = numpy.dot(P.T, DTE_origin)
                    print("PCA-" + str(10 - i) + ":")
                for G in Gs:
                    print("G:", G)
                    S = generate_GMM_scores(DTR, LTR, DTE, [G, G_auth], 0.01, 0.1, 1e-6, diagonal, tied_covariance)
                    minDCF = compute_min_DCF(p_eff, 1, 1, S, LTE)
                    print("minDCF:" + str(minDCF))
                    minDCFs.append(minDCF)
                    if minDCF < best_min:
                        best_min = minDCF
                        best_params = (G, G_auth, 10 - i, diagonal, tied_covariance)
                    tot_minDCFs.append(minDCF)
                PCA_minDCFs.append(minDCFs)
        print("diagonal:", diagonal, "tied_covariance:", tied_covariance)
        print("best params", best_params)
        print("best minDCF", best_min)
        best_min = 1000
        best_params = ()

    # # write PCA_minDCFs to txt file
    # with open(path_name + ".txt", "w") as f:
    #     for line in PCA_minDCFs:
    #         f.write(str(line) + "\n")
    #     f.write("best minDCF: " + str(min(tot_minDCFs)) + "\n")
    #     f.close()
    #
    # plt.figure()
    # plt.ylabel("minDCF")
    # plt.xlabel("G (for spoofed class)")
    # x_values = range(len(Gs))
    # plt.xticks(x_values, Gs)
    #
    # # recreate PCA_minDCFS, reading from the txt file
    # PCA_minDCFs = []
    # with open(path_name + ".txt", "r") as f:
    #     lines = f.readlines()
    #     for line in lines[:-1]:
    #         line = line.strip("[").strip("]").strip("\n").split(",")
    #         line[-1] = line[-1].strip("]")
    #         line = [float(i) for i in line]
    #         PCA_minDCFs.append(line)
    #     best_minDCF = float(lines[-1].split(":")[-1].strip("\n"))
    #     best_minDCF = round(best_minDCF, 3)
    #     f.close()
    # plt.title("authentic G=" + str(G_auth) + " (best minDCF: " + str(best_minDCF) + ")")
    # for i in range(len(PCA_minDCFs)):
    #     plt.plot(x_values, PCA_minDCFs[i], label="PCA-" + str(10 - i) if i != 0 else "No-PCA")
    #
    # plt.legend()
    # plt.grid()
    # plt.savefig(path_name + ".png")
    # plt.show()


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

    return S


if __name__ == '__main__':
    start_time = time.perf_counter()
    # GMM_evaluation_on_test()
    GMM_validation_on_test()
    # best_model_bayes_error_plot()
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print("Execution time:", execution_time, "seconds")
