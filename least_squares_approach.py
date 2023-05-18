import numpy as np
from scipy.optimize import nnls, lsq_linear, OptimizeResult
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
from multiprocessing import Pool
from mosek.fusion import *
from itertools import product


def convex_problem(args):
    """
    Function for evaluating the convex optimization to obtain lambda0, lambda1
    :param data:    list of np arrays phi1, phi2, and psi
    :return:
    """
    phi1, phi2, psi = args[0], args[1], args[2]

    # Flatten the arrays
    phi1 = phi1.reshape(-1) / phi1.sum()
    phi2 = phi2.reshape(-1) / phi2.sum()
    psi = psi.reshape(-1) / psi.sum()

    # Create A array for problem
    A = np.vstack([phi1, phi2]).T
    A = np.ascontiguousarray(A)

    # Try MOSEK
    M = lse(A, psi)
    M.solve()
    return M.getVariable("w").level()


def lse(X, y):
    n, d = len(X), len(X[0])
    M = Model("LSE")

    # the regression coefficients
    w = M.variable("w", d)

    # the bound on the norm of the residual
    t = M.variable("t")
    r = Expr.sub(y, Expr.mul(X, w))
    # t \geq |r|^2
    M.constraint(Expr.vstack(t, r), Domain.inQCone())
    M.constraint(w, Domain.greaterThan(0.0))
    M.constraint(w, Domain.lessThan(1.0))
    M.constraint(Expr.sum(w), Domain.equalsTo(1.0))

    M.objective(ObjectiveSense.Minimize, t)

    return M


plt.rcParams['figure.autolayout'] = True

if __name__ == "__main__":
    # Define paths to use for individual beams psi1 and psi2 and combined
    phi1_path = "Discord images_1100_1001_19 June/phi1/"
    phi1_suffixes = ["1_20.txt", "1_40.txt", "1_60.txt", "1_80.txt", "1_100.txt", "1_120.txt", "1_140.txt", "1_160.txt", "1_180.txt"]
    phi2_path = "Discord images_1100_1001_19 June/phi2/"
    phi2_suffixes = ["2_20.txt", "2_40.txt", "2_60.txt", "2_80.txt", "2_100.txt", "2_120.txt", "2_140.txt", "2_160.txt", "2_180.txt"]
    comb_path = "Discord images_1100_1001_19 June/psi/"
    comb_suffixes = ["12_20.txt", "12_40.txt", "12_60.txt", "12_80.txt", "12_100.txt", "12_120.txt", "12_140.txt", "12_160.txt", "12_180.txt"]

    # Create column headings for reading in data
    cols = tuple(i for i in range(768))

    # Loop over data and calculate lambda0 and lambda 1 from experimental data
    phi1s = [np.genfromtxt(phi1_path + _, delimiter=';', skip_header=8, usecols=cols) for _ in phi1_suffixes]
    phi2s = [np.genfromtxt(phi2_path + _, delimiter=';', skip_header=8, usecols=cols) for _ in phi2_suffixes]
    psis = [np.genfromtxt(comb_path + _, delimiter=';', skip_header=8, usecols=cols) for _ in comb_suffixes]

    with Pool() as pool:
        args = zip(phi1s, phi2s, psis)
        ans = np.array(pool.map(convex_problem, args))

    print(ans)
    lambda0 = ans.T[:][0]
    lambda1 = ans.T[:][1]

    # Need to normalize eigenvalues
    lambda0 = np.array(lambda0)
    lambda1 = np.array(lambda1)

    # Get intensities to calculate the expected lambdas
    I1 = np.array([27.2, 49.5, 78.1, 107, 133, 160, 159, 164, 161])
    I2 = np.array([136, 127, 130, 132, 137, 135, 113, 103, 90.3])
    exp_lambda0 = I1 / (I1 + I2)
    exp_lambda1 = I2 / (I1 + I2)

    print("exp_lambda0s: ", exp_lambda0)
    print("exp_lambda1s: ", exp_lambda1)
    print("exp_lambda_tot: ", exp_lambda0 + exp_lambda1)


    labels = ["20", "40", "60", "80", "100", "120", "140", "160", "180"]
    x = np.arange(I1.shape[0])
    width = 0.35
    fig1, ax1 = plt.subplots()
    ax1.set_title("Lambda0")
    rect1 = ax1.bar(x - width/2, lambda0, width, label='Simulated')
    rect2 = ax1.bar(x + width/2, exp_lambda0, width, label='Expected')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("File Used")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.set_title("Lambda1")
    rect3 = ax2.bar(x - width/2, lambda1, width, label='Simulated')
    rect4 = ax2.bar(x + width/2, exp_lambda1, width, label='Expected')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("File Used")
    ax2.legend()

    plt.figure(3)
    plt.imshow(lambda0[4]*phi1s[4] + lambda1[4]*phi2s[4])
    plt.colorbar()
    plt.title("Simulated")

    plt.figure(4)
    plt.imshow(psis[0])
    plt.colorbar()
    plt.title("Experimental")

    # Output lambda0s and lambda1s to npz file for later use
    outfile = "lamdas_from_mosek.npz"
    np.savez(outfile,
             lambda0=lambda0,
             lambda1=lambda1,
             )

    plt.show()
