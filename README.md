# pinn-episodic-source-transport

Code for the manuscript

> **Physics-informed neural networks for solute transport in porous media with an episodic source: error sources and identification of switching times**
> M. K. Singh, C. Roy Laskar. Submitted to *Computers & Geosciences*.

The code solves the one-dimensional advection–dispersion–reaction equation with an intermittent (episodic) inlet source using physics-informed neural networks (PINNs). It compares seven PINN variants over three Péclet regimes and twenty random seeds, and it recovers the source reactivation time from noisy downstream observations (inverse problem).

## Requirements

- Python 3.10 or later
- JAX, Optax, NumPy, mpmath, Matplotlib

```bash
pip install -r requirements.txt
```

A GPU is recommended for the forward study. Everything also runs on a CPU, only more slowly.

## Files

| File | Purpose |
|---|---|
| `params.py` | All physical, numerical and training parameters used in the study |
| `reference.py` | Reference solutions: stabilised Laplace-domain solution (de Hoog inversion) and explicit finite-difference scheme with a second-order ghost-node outlet |
| `pinn.py` | The seven PINN variants (JAX), trained for all seeds at once with `vmap` |
| `train.py` | Forward-study driver: job list, parameter sweeps, checkpointing and resume |
| `inverse.py` | Inverse problem: recovery of the reactivation time `T_off` from noisy wells |
| `analyze.py` | Builds every table and figure from the saved results |

## Reproducing the results

Run the steps in order from the repository folder.

```bash
# 1. Reference solutions and grid convergence (Table 2, Figs 2-3)
python analyze.py --stage reference

# 2. Forward study: 7 variants x 3 Peclet numbers, beta_w sweep,
#    lift study and inlet-ramp study (20 seeds each)
python train.py --sweep all

# 3. Inverse problem: switching-time identification
python inverse.py

# 4. Tables and figures
python analyze.py --stage pinn
python analyze.py --stage corner
python analyze.py --stage inverse
```

Results are written to `results/` and `results_inverse/`, cached reference fields to `cache/`, and tables and figures to `out/tables/` and `out/figures/`.

### Useful options

```bash
python train.py --list                        # show the job list
python train.py --sweep all --smoke           # quick test: 300 epochs, 2 seeds
python train.py --sweep all --workers 2 --worker 0   # run half of the jobs
python train.py --sweep all --workers 2 --worker 1   # ...and the other half
python inverse.py --smoke                     # quick test of the inverse problem
```

Training checkpoints are written every 1000 epochs. An interrupted job resumes from its last checkpoint, and finished jobs are skipped. Smoke-test results are for checking the installation only.

## Output files and the manuscript

| Manuscript | Output file |
|---|---|
| Table 2 | `out/tables/table2_grid.tex` |
| Table 3 | `out/tables/table3_ablation.tex` |
| Table 4 | `out/tables/table4_main.tex` |
| Table 5 | `out/tables/table8_corner.tex` |
| Table 6 | `out/tables/table6_paired.tex` |
| Table 7 | `out/tables/betaw.json` |
| Table 8 | `out/tables/lift.json` |
| Table 9 | `out/tables/table7_inverse.tex` |
| Fig. 2 | `out/figures/fig01_problem.pdf` |
| Fig. 3 | `out/figures/fig02_verification.pdf` |
| Fig. 4 | `out/figures/fig05_temporal.pdf` |
| Fig. 5 | `out/figures/fig06_triplets.pdf` |
| Fig. 6 | `out/figures/fig15_corner_ramp.pdf` |
| Fig. 7 | `out/figures/fig07_distributions.pdf` |
| Fig. 8 | `out/figures/fig08_paired.pdf` |
| Fig. 9 | `out/figures/fig09_betaw.pdf` |
| Fig. 10 | `out/figures/fig12_lift.pdf` |
| Fig. 11 | `out/figures/figS1_residual.pdf` |
| Fig. 12 | `out/figures/fig13_inverse_toff.pdf` |
| Fig. 13 | `out/figures/fig14_inverse_convergence.pdf` |

`analyze.py` also writes some additional diagnostic figures (profiles, parity plots, loss histories and training cost) that are not shown in the manuscript. Key numbers quoted in the text are collected in `out/tables/inserts.json`.

## Citation

If you use this code, please cite the paper (full reference to be added on publication):

> Singh, M.K., Roy Laskar, C. Physics-informed neural networks for solute transport in porous media with an episodic source: error sources and identification of switching times. *Computers & Geosciences* (submitted).

## Licence

MIT. See `LICENSE`.

## Contact

Chinmoy Roy Laskar, Department of Mathematics and Computing, Indian Institute of Technology (Indian School of Mines) Dhanbad, India. Email: crlaskar99@gmail.com
