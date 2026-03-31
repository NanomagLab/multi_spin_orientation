# Numerical Simulation of Topological Point Defects in Non-Abelian Order-Parameter Spaces Realized by Antiferromagnetically Coupled Heisenberg Spins

Code supplement for:

**Numerical Simulation of Topological Point Defects in Non-Abelian Order-Parameter Spaces Realized by Antiferromagnetically Coupled Heisenberg Spins**

Multi-spin dynamics on a triangular lattice, implemented with **TensorFlow**.  
Figure scripts live under [`scripts/`](scripts/); outputs are written under `figures/figure*` (created when you run the scripts).

## Citation

When you use this repository, please cite the paper (add volume/pages/DOI once available):

```bibtex
@article{YOURKEY,
  title   = {Numerical Simulation of Topological Point Defects in Non-Abelian Order-Parameter Spaces Realized by Antiferromagnetically Coupled Heisenberg Spins},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## Environment

Dependencies are pinned in [`environment.yml`](environment.yml): Conda (Python 3.9, CUDA/cuDNN, NumPy, tqdm, Mayavi, VTK, Plotly) plus a **`pip:`** block for **TensorFlow** and **Kaleido** (static PNG export for Plotly).

```bash
conda env create -f environment.yml
conda activate non_abelian
```

The `pip:` entries run automatically during `conda env create`; no separate `pip install` step is required.

## Running

Run all commands from the **repository root** so package imports resolve (`lib`, `scripts`).

**Batch (all figures).** [`main.py`](main.py) invokes each figure script in a fixed order via `python -m scripts.<module>`. To change the order or skip a script, edit the `FIGURE_SCRIPTS` list in `main.py`.

```bash
python main.py
```

**Single figure.** Run the desired module with `python -m scripts.<name>` (no `.py` suffix). For example, to reproduce only the content of [`scripts/figure2b.py`](scripts/figure2b.py):

```bash
python -m scripts.figure2b
```

Running `python scripts/figure2b.py` directly may fail to resolve `import lib...` because the project root is not on `sys.path`.

Available figure modules (same set and order as [`main.py`](main.py) `FIGURE_SCRIPTS`):

| Module | File |
|--------|------|
| `scripts.figure2a` | [`scripts/figure2a.py`](scripts/figure2a.py) |
| `scripts.figure2b` | [`scripts/figure2b.py`](scripts/figure2b.py) |
| `scripts.figure3a` | [`scripts/figure3a.py`](scripts/figure3a.py) |
| `scripts.figure3b` | [`scripts/figure3b.py`](scripts/figure3b.py) |
| `scripts.figure3c` | [`scripts/figure3c.py`](scripts/figure3c.py) |
| `scripts.figure4a` | [`scripts/figure4a.py`](scripts/figure4a.py) |
| `scripts.figure4b` | [`scripts/figure4b.py`](scripts/figure4b.py) |
| `scripts.figure4c` | [`scripts/figure4c.py`](scripts/figure4c.py) |
| `scripts.figure5a` | [`scripts/figure5a.py`](scripts/figure5a.py) |
| `scripts.figure5b` | [`scripts/figure5b.py`](scripts/figure5b.py) |
| `scripts.figure5c` | [`scripts/figure5c.py`](scripts/figure5c.py) |

### Reproducibility (random seeds)

Scripts that use TensorFlow / NumPy randomness set seeds in their `if __name__ == "__main__":` block (e.g. `random`, `numpy`, `tf.random`). For bitwise-identical runs, use the same conda env and TensorFlow version as in `environment.yml`.

## Layout

| Path | Role |
|------|------|
| [`scripts/figure2a.py`](scripts/figure2a.py) … [`scripts/figure5c.py`](scripts/figure5c.py) | Entry points for paper figures |
| [`main.py`](main.py) | Batch runner: all figures in order (`python -m scripts.figure*`) |
| [`lib/magnetism.py`](lib/magnetism.py) | Effective field, energy density, spin updates |
| [`lib/spin_utils.py`](lib/spin_utils.py) | Grid, group algebra, defect classification, masks, quaternion helpers |
| [`lib/initial_conditions.py`](lib/initial_conditions.py) | Initial spin fields (NumPy where appropriate; TF for dynamics hooks) |
| [`lib/trajectory.py`](lib/trajectory.py) | Trajectory extraction from defect records |
| [`lib/visualize.py`](lib/visualize.py) | Spin plots (Mayavi), 3D trajectories (Plotly) |
| `figures/` | Generated outputs (gitignored by default; see [`.gitignore`](.gitignore)) |
