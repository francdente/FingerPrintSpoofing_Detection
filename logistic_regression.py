import matplotlib.pyplot as plt
import numpy
import scipy
import time
from utils import *


def compute_accuracy(LP, LTE):
    res = LP == LTE
    correctPredictions = res.sum(0)
    acc = correctPredictions / res.size
    err = 1 - acc
    return acc, err


def best_model_bayes_error_plot():
    """Bayes error plot of our chosen model"""
    DTR, LTR = load("Train.txt")
    DTR = center_Dataset(DTR)
    P = calculate_pca_proj_matrix(DTR, 6)
    DTR = numpy.dot(P.T, DTR)
    DTR = to_expanded_feature_space(DTR)
    DTR = center_Dataset(DTR)

    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    print(len(LTR[LTR == 1]))
    print(len(LTR[LTR == 0]))
    # p_t=0.091, lambda=0.1, PCA-6
    p_t = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    S, L = k_fold_validation_logreg(DTR, LTR, 5, 0.1, p_t, split_db_k_to_1_v2)
    # recover llrs from LR scores
    S = S - numpy.log(p_t / (1 - p_t))
    # effPriorLogOdds = numpy.linspace(-3, 3, 100)
    # print_bayes_error_plot(effPriorLogOdds, S, L, "calibration/Q-LR_b_error.png", save=True)

    print("minDCF:" + str(compute_min_DCF(p_eff, 1, 1, S, L)))
    predictedLabels = compute_predictions(S, - numpy.log(p_eff / (1 - p_eff)))
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


def logreg_evaluation_on_test():
    """Evaluation of a chosen model trained on the whole training set on the test set."""
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

    S = compute_logreg_scores_with_priors(DTR, LTR, DTE, 0.1, p_eff)
    # recover llrs from scores
    S = S - numpy.log(p_eff / (1 - p_eff))

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


def logreg_validation_on_test():
    """
    Validation on evaluation set to find the configuration that would have been optimal
    """
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = center_Dataset(DTR)
    DTE = center_Dataset(DTE)
    DTR_origin = DTR
    DTE_origin = DTE
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    lambdas = [10 ** 2, 10, 1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    lambdas = lambdas[::-1]
    PCA_dcfs = []
    p_t = 0.344
    for p_t in [0.048, 0.091, 0.167, 0.344]:
        best_min = 1000
        best_params = ()
        for i in range(5):
            if i == 0:
                print("NO-PCA")
            else:
                print("PCA-" + str(10 - i))
                P = calculate_pca_proj_matrix(DTR_origin, 10 - i)
                DTR = numpy.dot(P.T, DTR_origin)
                DTE = numpy.dot(P.T, DTE_origin)
            DTR = to_expanded_feature_space(DTR)
            DTE = to_expanded_feature_space(DTE)
            DTR = center_Dataset(DTR)
            DTE = center_Dataset(DTE)
            # DTR = z_normalization(DTR)
            # DTE = z_normalization(DTE)
            DCFs = []
            for l in lambdas:
                print("lambda:" + str(l))
                S = compute_logreg_scores_with_priors(DTR, LTR, DTE, l, p_t)
                # recover llrs from scores
                S = S - numpy.log(p_eff / (1 - p_eff))
                minDCF = compute_min_DCF(p_eff, 1, 1, S, LTE)
                DCFs.append(minDCF)
                print("minDCF:" + str(minDCF))
                if minDCF < best_min:
                    best_min = minDCF
                    best_params = (10 - i, l)
            PCA_dcfs.append(DCFs)
        print("best minDCF:" + str(best_min))
        print("best params:" + str(best_params))

        name = "evaluation/test_valid_QLR_p=" + str(p_t)
        # write gamma_minDCFs to txt file
        with open(name + ".txt", "w") as f:
            for line in PCA_dcfs:
                f.write(str(line) + "\n")
            f.close()

        plt.figure()
        plt.xlabel("lambda")
        plt.ylabel("minDCF")
        x_values = range(len(lambdas))
        plt.xticks(x_values, lambdas)
        plt.ylim(0.22, 0.50)

        # recreate PCA_iDCFS, reading from the txt file
        PCA_dcfs = []
        with open(name + ".txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("[").strip("]").strip("\n").split(",")
                line[-1] = line[-1].strip("]")
                line = [float(i) for i in line]
                PCA_dcfs.append(line)
            f.close()

        for i in range(5):
            plt.plot(x_values, PCA_dcfs[i], label="PCA=" + str(10 - i) if i != 0 else "NO-PCA")
        plt.grid()
        plt.title("bestMinDCF=" + str(round(best_min, 3)) + " lambda=" + str(best_params[1]) + " PCA=" + str(best_params[0]))
        plt.legend()
        plt.savefig(name + ".png")
        plt.show()


def logreg_validation_on_train():
    """
    Used to find the optimal configuration for our Q-LogReg using validation set extracted from training set
    """
    DTR, LTR = load("Train.txt")
    p_eff = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 10)
    p_eff2 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 20)
    p_eff3 = (0.5 * 1) / (0.5 * 1 + (1 - 0.5) * 5)
    DTR_origin = DTR
    DTR_origin = center_Dataset(DTR_origin)
    lambdas = [10, 1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4), 10 ** (-5)]
    lambdas = lambdas[::-1]
    PCA_minDCFs = []
    for i in range(5):
        if i == 0:
            print("NO-PCA")
            DTR = DTR_origin
        else:
            print("PCA-" + str(10 - i))
            P = calculate_pca_proj_matrix(DTR_origin, 10 - i)
            DTR = numpy.dot(P.T, DTR_origin)
        DTR = to_expanded_feature_space(DTR)
        DTR = center_Dataset(DTR)
        minDCFs = []
        for l in lambdas:
            print("lambda:" + str(l))
            S, L = k_fold_validation_logreg(DTR, LTR, 5, l, p_eff, split_db_k_to_1_v2)
            # recover llrs from scores
            S = S - numpy.log(p_eff / (1 - p_eff))
            minDCF = compute_min_DCF(p_eff, 1, 1, S, L)
            minDCFs.append(minDCF)
            print("minDCF:" + str(minDCF))
        PCA_minDCFs.append(minDCFs)
        print(PCA_minDCFs)

    # write gamma_minDCFs to txt file
    with open("lr/Q-LogReg_znorm_p=0.091V2.txt", "w") as f:
        for line in PCA_minDCFs:
            f.write(str(line) + "\n")
        f.close()

    plt.figure()
    plt.xlabel("lambda")
    plt.ylabel("minDCF")
    x_values = range(len(lambdas))
    plt.xticks(x_values, lambdas)
    # limit the y-axis up to 0.55
    plt.ylim(0.22, 0.55)

    # recreate PCA_iDCFS, reading from the txt file
    PCA_minDCFs = []
    with open("lr/Q-LogReg_znorm_p=0.091V2.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("[").strip("]").strip("\n").split(",")
            line[-1] = line[-1].strip("]")
            line = [float(i) for i in line]
            PCA_minDCFs.append(line)
        f.close()

    # find best minDCF
    mins = []
    for i in range(len(PCA_minDCFs)):
        mins.append(min(PCA_minDCFs[i]))
    best_min = min(mins)

    for i in range(5):
        plt.plot(x_values, PCA_minDCFs[i], label="PCA=" + str(10 - i) if i != 0 else "NO-PCA")

    plt.grid()
    plt.title("bestMinDCF=" + str(round(best_min, 3)))
    plt.legend()
    plt.savefig("lr/Q-LogReg_znorm_p=0.091V2.png")
    plt.show()



if __name__ == '__main__':
    start_time = time.perf_counter()
    DTR, LTR = load("Train.txt")
    # best_model_bayes_error_plot()
    # logreg_evaluation_on_test()
    # best_model_bayes_error_plot()
    # logreg_evaluation_on_test()
    # logreg_validation_on_train()
    # logreg_validation_on_test()
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print("Execution time:", execution_time, "seconds")
