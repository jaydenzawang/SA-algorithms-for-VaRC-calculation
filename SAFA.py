#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 19:41:10 2025

@author: jaydenwang
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, beta
import time
from tqdm import tqdm


np.random.seed(42)


# Parameters
N = 10  # Portfolio size
tau = 0.5  # Correlation between Z_D and Z_L
rho_D = np.sqrt(0.5) * np.ones(N)  
rho_L = np.sqrt(0.5) * np.ones(N)  
alpha_beta = 2  
beta_beta = 5   
p = np.array([0.01 * (i+1) for i in range(N)]) # Unconditional PDs: 0.01 to 0.10
x_d = norm.ppf(1 - p) # Default thresholds


# Given VaR values for alphas [0.95,0.96,0.97,0.98,0.99]
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]
a_VaR_values = [1.1432, 1.3098, 1.5397, 1.8733, 2.4625] 


# # Demo For testing with a single value
# a_VaR_values = [1.1432]


# Simulation parameters (modifiable)
N_outer = 1000 # Number of outer simulations for Z (e.g., 1000 for testing, increase for accuracy)
N_inner = 1000 # Number of inner simulations for L_{-i} and epsilon_i (e.g., 10000)
reps = 10 # Number of repetitions for averaging and SE


def compute_safa_varc(a_VaR_values, N_outer, N_inner, reps, u=None):

    cov_matrix = np.array([[1, tau], [tau, 1]])
    u = np.ones(N) if u is None else np.asarray(u, dtype=float) # shape (N,)

    results = []


    def f_cond_scalar(e, z_L):
        e = float(np.clip(e, 1e-12, 1-1e-12))
        v = norm.ppf(beta.cdf(e, alpha_beta, beta_beta))
        s = np.sqrt(1 - rho_L[0]**2)
        w = (v - rho_L[0]*z_L) / s
        phi_v = norm.pdf(v)
        if phi_v <= 0:
            return 0.0
        return beta.pdf(e, alpha_beta, beta_beta) * norm.pdf(w) / (phi_v * s)

    for a in a_VaR_values:
        VaRC_reps = []

        total_start = time.time()

        for _ in tqdm(range(reps), desc="Replications"):
            A = np.zeros(N)

            for _ in range(N_outer):
                # Sample common factors
                z_L, z_D = np.random.multivariate_normal([0, 0], cov_matrix)

                # Inner idiosyncratics
                eta_L = np.random.randn(N, N_inner)
                eta_D = np.random.randn(N, N_inner)

                # LGD path for all obligors
                Y = rho_L[:, None]*z_L + np.sqrt(1 - rho_L[:, None]**2)*eta_L
                U = norm.cdf(Y)
                epsilon_all = beta.ppf(U, alpha_beta, beta_beta)  # (N, N_inner)

                # Default indicators
                X = rho_D[:, None]*z_D + np.sqrt(1 - rho_D[:, None]**2)*eta_D
                D = (X > x_d[:, None]).astype(float)

                # Individual losses Li = u_i * epsilon_i * D_i
                losses = (u[:, None]) * epsilon_all * D   # (N, N_inner)

                # Portfolio loss and L_{-i}
                L_total = np.sum(losses, axis=0)
                L_minus = L_total[None, :] - losses

                # Sort for binary search (per obligor)
                sorted_L_minus = np.sort(L_minus, axis=1)  # (N, N_inner)

                # Sample epsilon_i | z_L (shared since rho_L identical vector)
                eta_L_i = np.random.randn(N_inner)
                s = np.sqrt(1 - rho_L[0]**2)
                Y_i = rho_L[0]*z_L + s*eta_L_i
                U_i = norm.cdf(Y_i)
                epsilon_samples = np.clip(beta.ppf(U_i, alpha_beta, beta_beta), 1e-9, 1-1e-9)

                # f_cond, f'_cond, g (vectorized at epsilon_samples)
                f_beta = beta.pdf(epsilon_samples, alpha_beta, beta_beta)
                u_cdf = beta.cdf(epsilon_samples, alpha_beta, beta_beta)
                v = norm.ppf(u_cdf)
                w = (v - rho_L[0]*z_L) / s
                phi_v = norm.pdf(v)
                phi_w = norm.pdf(w)

                f_cond = np.zeros(N_inner)
                mask_phi = phi_v > 0
                f_cond[mask_phi] = f_beta[mask_phi] * phi_w[mask_phi] / (phi_v[mask_phi] * s)

                term1 = np.zeros(N_inner)
                term1[mask_phi] = f_beta[mask_phi] / phi_v[mask_phi] * (v[mask_phi] - w[mask_phi] / s)
                term2 = (alpha_beta - 1) / epsilon_samples - (beta_beta - 1) / (1 - epsilon_samples)
                f_prime_cond = f_cond * (term1 + term2)

                g = np.zeros(N_inner)
                mask_f = f_cond > 0
                g[mask_f] = 1 + epsilon_samples[mask_f] * (f_prime_cond[mask_f] / f_cond[mask_f])

                # Conditional default probs p_i(z_D)
                arg = (x_d - rho_D * z_D) / np.sqrt(1 - rho_D**2)
                p_cond = 1 - norm.cdf(arg)

                # Boundary terms with EAD:
                boundary = np.zeros(N)
                prod_all = np.prod(1 - p_cond)
                for i in range(N):
                    # Only when a / u_i < 1, the upper boundary contributes
                    if a > 0 and u[i] > 0 and (a / u[i]) < 1:
                        ei = a / u[i]
                        f_at = f_cond_scalar(ei, z_L)
                        denom = (1 - p_cond[i])
                        prod_i = prod_all / denom if denom > 0 else 0.0
                        # - a * f_cond(a/u_i) * prod_{j≠i}(1 - p_j)
                        boundary[i] = - a / u[i] * f_at * prod_i
                    # Else, boundary[i] stays 0 (includes the typical min(1, a/u_i)=1 case)

                inner_values = np.zeros(N)

                # F(a - u_i * epsilon)
                for i in range(N):
                    thresholds = a - u[i] * epsilon_samples
                    F = np.searchsorted(sorted_L_minus[i], thresholds, side='right') / N_inner
                    inner_avg = np.sum(F * g) / N_inner + boundary[i]
                    inner_values[i] = inner_avg

                # A_i = E_Z[ p_i(Z_D) * inner(z) ]
                A += (p_cond * inner_values)

            # Outer
            A = A / N_outer

            # Denominator via full allocation: f_L(a) = sum_i A_i / a
            sum_A = np.sum(A)
            f_L_a = sum_A / a if a != 0 else 0.0
            # print(f_L_a)

            VaRC = A / f_L_a if f_L_a != 0 else np.zeros(N)
            VaRC_reps.append(VaRC)

        total_time = time.time() - total_start

        VaRC_array = np.array(VaRC_reps)
        results.append({
            'a': a,
            'mean_VaRC': np.mean(VaRC_array, axis=0),
            'se_VaRC': np.std(VaRC_array, axis=0) / np.sqrt(reps),
            'rep_times': [],
            'total_time': total_time
        })

    return results


# Run the computation
safa_results = compute_safa_varc(a_VaR_values, N_outer, N_inner, reps) 


# Print final results
VaRCs = [r['mean_VaRC'] for r in safa_results]      
VaRC_SEs = [r['se_VaRC'] for r in safa_results]     
CPUs = [r['total_time'] for r in safa_results] 


Risk_Contributions = pd.DataFrame({
    'VaRC 0.95' : VaRCs[0],
    'VaRC S.E. 0.95' : VaRC_SEs[0],
    'VaRC 0.95 CPU' : CPUs[0],
    
    'VaRC 0.96' : VaRCs[1],
    'VaRC S.E. 0.96' : VaRC_SEs[1],
    'VaRC 0.96 CPU' : CPUs[1],
    
    'VaRC 0.97' : VaRCs[2],
    'VaRC S.E. 0.97' : VaRC_SEs[2],
    'VaRC 0.97 CPU' : CPUs[2],
    
    'VaRC 0.98' : VaRCs[3],
    'VaRC S.E. 0.98' : VaRC_SEs[3],
    'VaRC 0.98 CPU' : CPUs[3],
    
    'VaRC 0.99' : VaRCs[4],
    'VaRC S.E. 0.99' : VaRC_SEs[4],
    'VaRC 0.99 CPU' : CPUs[4],

    }, index=pd.Index([f'Obligor {i+1}' for i in range(N)])).T

Risk_Contributions = Risk_Contributions.round(4)
# Risk_Contributions.to_csv('VaRC SAFA.csv')
print(Risk_Contributions)
