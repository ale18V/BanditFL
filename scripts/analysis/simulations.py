import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from scipy.stats import hypergeom
from tqdm import tqdm, trange

import seaborn as sns
sns.set_theme()


def KL_ber(alpha,beta):
    # KL divergence of Bernoulli variables
    if alpha == 0:
        return beta==0
    if alpha==1:
        return b==1
    return alpha * np.log(alpha/beta) + (1-alpha)*np.log((1-alpha)/(1-beta))

def sat(n,b,s,b_hat,T,p):
    # Returns true if the theoretical condition is satisfied 
    D = KL_ber(b_hat/s, b/(n-1))
    if D==0:
        return s==n
    return s >= min(1/D * np.log((n-b)*T/(1-p)) , n-1)


def n_th(n,b,T,p):
    # Returns the theoretical number of samples needed to satisfy Lemma 4.2

    return math.ceil( max(1/(1/2 - (b/n))**2, 3/(b/n) ) * np.log(4*(n-b)*T/ (1-p))) +2
    #return math.ceil( 1/ KL_ber(1/2, b/n) * np.log(4*n*T/ (1-p))) +2

def simulate(n,b,s, T, mult = 1):
    rng = np.random.default_rng()
    return max(rng.hypergeometric(b, n-b, s, n*T*mult))
    #return max(np.random.Generator.hypergeometric(b, n-b, s, n*T*mult))
    #return max(np.random.hypergeometric(b, n-b, s, n*T*mult))







n = 1000

T = 200
mult = 5
s_max = 60
s_min = 10
for b in trange(20, 120, 20):
    max_fraction = []
    for s in range(s_min, s_max, 10):
        fractions = []
        for _ in range(mult):
            b_hat = simulate(n, b, s, T, 1)
            fractions.append(b_hat / (s+1))
        mean_fraction = np.mean(fractions)
        std_fraction = np.std(fractions)
        max_fraction.append((mean_fraction, std_fraction))


    erplt = plt.errorbar(range(s_min, s_max, 10), [x[0] for x in max_fraction], yerr=[x[1] for x in max_fraction], fmt='-o', label = f"b={b}")
    color = erplt.lines[0].get_color()
    upper_bound = [x[0] + x[1] for x in max_fraction]
    lower_bound = [x[0] - x[1] for x in max_fraction]
    plt.fill_between(range(s_min, s_max, 10), lower_bound, upper_bound, color=color, alpha=0.1)

    plt.xlabel("Number of neighbors")
    plt.ylabel(r"$\frac{\hat{b}}{s+1}$", rotation=0)
    plt.ylim(0,1)
    #plt.title(rf"Effect of $s$ on the Effective Byzantine Fraction, $n = {n}, b={b}$")
    plt.axhline(y=0.5, color='r', linestyle='--')
    #plt.savefig(f"t_b_hat/b_hat_n_{n}_b_{b}.png")
    #plt.clf()
   
plt.legend()
plt.title(rf"$n = {n}$")
# plt.title(rf"Effect of $s$ on the Effective Byzantine Fraction, $n = {n}$")
plt.savefig(f"exp_b_hat_new/b_hat_n_{n}.png")