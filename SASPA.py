#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 23:27:36 2025

@author: jaydenwang
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, beta
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import time
from tqdm import tqdm
from scipy.optimize import fsolve

np.random.seed(42)


# Parameters
N = 10  # Portfolio size
tau = 0.5  # Corr(Z_D, Z_L)
rho_D = np.sqrt(0.5) * np.ones(N)  # Factor loadings for default, uniform
rho_L = np.sqrt(0.5) * np.ones(N)  # Factor loadings for LGD, uniform
alpha_beta = 2  # Beta shape alpha
beta_beta  = 5  # Beta shape beta


# Unconditional PDs and default thresholds
p   = np.array([0.01 * (i + 1) for i in range(N)])  # 0.01 ... 0.10
x_d = norm.ppf(1 - p)


# VaR levels
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.1432, 1.3098, 1.5397, 1.8733, 2.4625]


# # Demo 
# a_VaR_values = [1.1432]


# Simulation / Quadrature Params
N_outer = 6000   # outer sims for Z
N_inner = 1000   # inner sims for epsilon_i & L_{-i}
reps    = 10     # repetitions for averaging and SE


# Quadrature orders
N_gh = 16  # Gauss-Hermite nodes per dim
N_gl = 16  # Gauss-Legendre nodes on [0,1]


# Numerical quadrature helpers
def gauss_legendre_on_01(n):
    x, w = leggauss(n)
    e = 0.5 * (x + 1.0)
    w = 0.5 * w
    return e, w


gh_x, gh_w = hermgauss(N_gh)  # not used directly; kept if need raw nodes/weights


# Conditional LGD density f_{eps | Z_L}(e | zL, rhoL)
def f_eps_cond(e, zL, rhoL, a_beta, b_beta):

    e = np.asarray(e, dtype=float)
    f_b = beta.pdf(e, a_beta, b_beta)
    u = beta.cdf(e, a_beta, b_beta)
    v = norm.ppf(u)
    s = np.sqrt(1.0 - rhoL**2)
    w = (v - rhoL * zL) / s
    num = norm.pdf(w)
    denom = norm.pdf(v) * s
    dens = np.where(denom > 1e-300, f_b * num / denom, 0.0)
    return dens


# M_eps and derivatives at t: M_k(t | Z_L) via Gauss-Legendre
def M_eps_and_derivs(t, zL, rhoL, a_beta, b_beta, n_gl=N_gl):
    e_nodes, e_weights = gauss_legendre_on_01(n_gl)
    fvals = f_eps_cond(e_nodes, zL, rhoL, a_beta, b_beta)
    et  = np.exp(t * e_nodes)
    M0 = np.sum(e_weights * et * fvals)
    M1 = np.sum(e_weights * (e_nodes) * et * fvals)
    M2 = np.sum(e_weights * (e_nodes**2)* et * fvals)
    M3 = np.sum(e_weights * (e_nodes**3)* et * fvals)
    M4 = np.sum(e_weights * (e_nodes**4)* et * fvals)
    return M0, M1, M2, M3, M4


# A_i and derivatives with EAD u_i
def A_i_and_derivs(t, zD, zL, rhoDi, rhoLi, xdi, a_beta, b_beta, ui):

    
    sD  = np.sqrt(1.0 - rhoDi**2)
    arg = (xdi - rhoDi * zD) / sD
    pz  = 1.0 - norm.cdf(arg)

    s   = t * ui
    M0, M1, M2, M3, M4 = M_eps_and_derivs(s, zL, rhoLi, a_beta, b_beta)

    ui1 = ui
    ui2 = ui * ui
    ui3 = ui2 * ui
    ui4 = ui2 * ui2

    A0 = 1.0 - pz + pz * M0
    A1 = pz * ui1 * M1
    A2 = pz * ui2 * M2
    A3 = pz * ui3 * M3
    A4 = pz * ui4 * M4
    return A0, A1, A2, A3, A4


# K(t) and derivatives (cumulant func) with EAD vector u_vec
def K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec):
    n = len(xds)
    A0s = np.empty(n)
    A1s = np.empty(n)
    A2s = np.empty(n)
    A3s = np.empty(n)
    A4s = np.empty(n)

    for i in range(n):
        A0, A1, A2, A3, A4 = A_i_and_derivs(
            t, zD, zL, rhoD_vec[i], rhoL_vec[i], xds[i], a_beta, b_beta, u_vec[i]
        )
        
        A0s[i] = A0
        A1s[i] = A1
        A2s[i] = A2
        A3s[i] = A3
        A4s[i] = A4

    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.nan_to_num(A1s / A0s)
        r2 = np.nan_to_num(A2s / A0s)
        r3 = np.nan_to_num(A3s / A0s)
        r4 = np.nan_to_num(A4s / A0s)

    K0 = np.sum(np.log(A0s))
    K1 = np.sum(r1)
    K2 = np.sum(r2 - r1**2)
    K3 = np.sum(r3 - 3 * r1 * r2 + 2 * r1**3)
    K4 = np.sum(r4 - 4 * r1 * r3 - 3 * r2**2 + 12 * (r1**2) * r2 - 6 * r1**4)
    return K0, K1, K2, K3, K4


# Solve saddlepoint K'(t) = a with u_vec using fsolve
def solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec,
                      t0=0.1, max_iter=200):

    def equation(t_array):

        t_val = float(t_array[0])
        try:
            _, K1, _, _, _ = K_and_derivs(
                t_val, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec
            )
            residual = K1 - a_target
        except (FloatingPointError, OverflowError, ValueError):
            residual = 1e6
        return np.array([residual], dtype=float)

    t_init = np.array([t0], dtype=float)

    root, info, ier, mesg = fsolve(
        equation,
        x0=t_init,
        full_output=True,
        maxfev=max_iter
    )

    t_hat = float(root[0])

    if ier != 1 or not np.isfinite(t_hat):
        t_hat = t0

    return t_hat


# Conditional SPA density f_{L|Z}(a|Z)
def spa_conditional_density(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec):
    t_hat = solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, t0=0.1)
    K0, _, K2, K3, K4 = K_and_derivs(t_hat, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
    if K2 <= 1e-12:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lam3 = np.nan_to_num(K3 / (K2**1.5))
        lam4 = np.nan_to_num(K4 / (K2**2))
    pref = np.exp(K0 - t_hat * a_target) / np.sqrt(2.0 * np.pi * K2)
    corr = 1.0 + 0.125 * (lam4 - (5.0 / 3.0) * (lam3**2))
    
    return max(0.0, pref * corr)


# Unconditional f_L(a) via 2D Gauss-Hermite
def f_L_via_SPA(a, tau, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, n_gh=N_gh):
    Sigma = np.array([[1.0, tau], [tau, 1.0]])
    Lch   = np.linalg.cholesky(Sigma)
    x_nodes, w_nodes = hermgauss(n_gh)
    total = 0.0
    factor = 1.0 / np.pi  # 2D GH: 1/(√π)^2

    for i in range(n_gh):
        for j in range(n_gh):
            xvec = np.array([x_nodes[i], x_nodes[j]])
            z = np.sqrt(2.0) * (Lch @ xvec)
            zL, zD = z[0], z[1]
            w = w_nodes[i] * w_nodes[j] * factor
            dens = spa_conditional_density(a, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
            if np.isfinite(dens):
                total += w * dens
    
    return total


# SASPA VaRC with EAD u_i, assumed to be unit
def compute_saspa_varc(a_VaR_values, N_outer, N_inner, reps, u=None):
    cov_matrix = np.array([[1, tau], [tau, 1]])
    u = np.ones(N) if u is None else np.asarray(u, dtype=float)

    results = []

    # helper: scalar f_{eps|ZL}(e | zL) for boundary term
    def f_cond_scalar(e, zL):
        e = float(e)
        fb = beta.pdf(e, alpha_beta, beta_beta)
        ua = beta.cdf(e, alpha_beta, beta_beta)
        v  = norm.ppf(ua)
        sL = np.sqrt(max(1 - rho_L[0]**2, 1e-14))
        w  = (v - rho_L[0] * zL) / sL
        phiv = norm.pdf(v)
        if phiv <= 0:
            return 0.0
        return fb * norm.pdf(w) / (phiv * sL)

    for a in a_VaR_values:
        VaRC_reps = []
        total_start = time.time()

        # Denominator f_L(a) via SPA with u_i
        t0 = time.time()
        f_L_a = f_L_via_SPA(a, tau, rho_D, rho_L, x_d, alpha_beta, beta_beta, u_vec=u, n_gh=N_gh)
        print(f"f_L(a) via SPA = {f_L_a:.6g} (once, {time.time()-t0:.2f}s)")

        for _ in tqdm(range(reps), desc="Replications"):
            A = np.zeros(N)

            # Outer loop for Z
            for _ in range(N_outer):
                # Sample Z = (Z_L, Z_D)
                Z = np.random.multivariate_normal([0, 0], cov_matrix)
                z_L, z_D = Z[0], Z[1]

                # Sample inner paths for all obligors
                eta_L = np.random.randn(N, N_inner)
                eta_D = np.random.randn(N, N_inner)

                # Compute Y_j, epsilon_j
                Y = rho_L[:, None] * z_L + np.sqrt(1 - rho_L[:, None]**2) * eta_L
                Uc = norm.cdf(Y)
                epsilon = beta.ppf(Uc, alpha_beta, beta_beta) # (N, N_inner)

                # Compute X_j, D_j
                X = rho_D[:, None] * z_D + np.sqrt(1 - rho_D[:, None]**2) * eta_D
                Dmat = (X > x_d[:, None]).astype(float) # (N, N_inner)

                # Individual losses with EAD
                losses = (u[:, None]) * epsilon * Dmat # (N, N_inner)

                # Total L and L_{-i}
                L_total = np.sum(losses, axis=0) # (N_inner,)
                L_minus = L_total[None, :] - losses # (N, N_inner)

                # Sort L_{-i} for each i for binary search
                sorted_L_minus = np.sort(L_minus, axis=1) # (N, N_inner)

                # Sample M = N_inner independent epsilon_i | z_L (shared rho_L)
                eta_L_i = np.random.randn(N_inner)
                sL = np.sqrt(1 - rho_L[0]**2)
                Y_i = rho_L[0] * z_L + sL * eta_L_i # rho_L same for all
                U_i = norm.cdf(Y_i)
                eps_i = beta.ppf(U_i, alpha_beta, beta_beta) # (N_inner,)

                # Vectorized computation for f_cond, f_prime_cond, g
                f_beta_vals = beta.pdf(eps_i, alpha_beta, beta_beta)
                u_vals = beta.cdf(eps_i, alpha_beta, beta_beta)
                v = norm.ppf(u_vals)
                w = (v - rho_L[0] * z_L) / sL
                phi_v = norm.pdf(v)
                phi_w = norm.pdf(w)
                f_cond = np.zeros(N_inner)
                mask_phi = phi_v > 0
                f_cond[mask_phi] = f_beta_vals[mask_phi] * phi_w[mask_phi] / (phi_v[mask_phi] * sL)

                term1 = np.zeros(N_inner)
                term1[mask_phi] = f_beta_vals[mask_phi] / phi_v[mask_phi] * (v[mask_phi] - w[mask_phi] / sL)
                term2 = (alpha_beta - 1) / eps_i - (beta_beta - 1) / (1 - eps_i)
                f_prime_cond = f_cond * (term1 + term2)

                g = np.zeros(N_inner)
                mask_f = f_cond > 0
                g[mask_f] = 1 + eps_i[mask_f] * (f_prime_cond[mask_f] / f_cond[mask_f])

                # Conditional default probabilities p_i(Z_D)
                arg = (x_d - rho_D * z_D) / np.sqrt(1 - rho_D**2)
                p_cond = 1 - norm.cdf(arg)

                # Boundary terms with EAD
                boundary = np.zeros(N)
                prod_all = np.prod(1 - p_cond)
                for i in range(N):
                    if a > 0 and u[i] > 0 and (a / u[i]) < 1:
                        ei = a / u[i]
                        f_at = f_cond_scalar(ei, z_L)
                        denom = (1 - p_cond[i])
                        prod_i = (prod_all / denom) if denom > 0 else 0.0
                        # - a * f_cond(a/u_i | z_L) * Π_{j≠i} (1 - p_j(Z_D))
                        boundary[i] = - a / u[i] * f_at * prod_i

                # inner averages with thresholds a - u_i * eps_i
                inner_values = np.zeros(N)
                for i in range(N):
                    thresholds = a - u[i] * eps_i
                    F = np.searchsorted(sorted_L_minus[i], thresholds, side='right') / N_inner
                    inner_avg = np.sum(F * g) / N_inner + boundary[i]
                    inner_values[i] = inner_avg

                # Accumulate A_i = E_Z[ p_i(Z_D) * inner(z) ]
                A += (p_cond * inner_values)

            # Average over outer loop
            A = A / N_outer

            # VaRC_i = A_i / f_L(a)
            VaRC = A / f_L_a if f_L_a != 0 else np.zeros(N)
            VaRC_reps.append(VaRC)

        total_time = time.time() - total_start
        VaRC_array = np.array(VaRC_reps)
        results.append({
            'a': a,
            'mean_VaRC': np.mean(VaRC_array, axis=0),
            'se_VaRC'  : np.std(VaRC_array, axis=0) / np.sqrt(reps),
            'rep_times': [],
            'total_time': total_time
        })

    return results


# Run the computation
if __name__ == "__main__":
    
    # Case 1: default u=None (equivalent to u_i = 1), identical to your current results
    saspa_results = compute_saspa_varc(a_VaR_values, N_outer, N_inner, reps, u=None)

    # # Case 2: specify EADs, but require recompute VaR values for this u 
    # u_vec = np.array([1.0, 1.2, 0.8, 2.0, 1.5, 1.0, 0.9, 1.3, 2.1, 0.7]) 
    # saspa_results = compute_saspa_varc(a_VaR_values, N_outer, N_inner, reps, u=u_vec)

    # Output table
    rows = []
    for r in saspa_results:
        a_val = r['a']
        mean_v = r['mean_VaRC']
        se_v   = r['se_VaRC']
        cpu    = r['total_time']

        rows.append((f'VaRC (a={a_val:.4f})', mean_v))
        rows.append((f'VaRC S.E. (a={a_val:.4f})', se_v))
        rows.append((f'CPU (a={a_val:.4f})', np.full_like(mean_v, cpu, dtype=float)))

    cols = [f'Obligor {i+1}' for i in range(N)]
    table_dict = {row_name: row_values for row_name, row_values in rows}
    Risk_Contributions = pd.DataFrame(table_dict, index=cols).T
    Risk_Contributions = Risk_Contributions.round(4)
    print(Risk_Contributions)
    # Risk_Contributions.to_csv('VaRC SASPA.csv')
