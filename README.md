# Simulation-Analytical Approach for Calculating VaR Contributions in Credit Portfolios

## Citation Requirement
If you use any code or algorithm (`SAFA.py`, `SASPA.py`, etc.) from this repository, **please cite our published paper**:

Wang, J. Z., Huang, Z., & Kwok, Y. K. (2026). Simulation-analytical approach for calculating VaR contributions in credit portfolios. *Quantitative Finance*. https://doi.org/10.1080/14697688.2026.2641764

Formats such as BibTeX/APA can be copied directly from the **Cite this repository** button on the right.

## Kep points
- $\text{VaRC}_{i,\alpha} = \frac{\mathbb{E}[L_i\delta(L-a)]}{\mathbb{E}[\delta(L-a)]}$
- Benchmark methods: `PMC.py` and `CE_RQMC.py`.
- Simulation-analytical algorithms: `SAFA.py` and `SASPA.py`.

## Files
- `PMC.py`: Plain Monte Carlo method.
- `CE_RQMC.py`: Iterative CE method enhanced with RQMC.
- `SAFA.py`: Estimates the denominator $\mathbb{E}[\delta(L-a)]$ via the Euler allocation principle.
- `SASPA.py`: Estimates the denominator $f_L(a)$ via SPA.
- `Loss_Threshold.py`: The loss threshold is involved in `CE_RQMC.py`.
- `VaRC results table.pdf`: The numerical results for our simulation-analytical algorithms and benchmark methods.
