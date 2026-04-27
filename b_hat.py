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
b = 10
s = 15

T = 100
print("Starting")
b_hat =simulate(n,b,s, T,10)
print(b_hat)
print(b_hat/s)


