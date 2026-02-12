# MC-SPF: Monte Carlo Spectral Photometric Fitter

MC-SPF is a Python-based tool for fitting Spectral Energy Distributions (SEDs) of galaxies using detailed stellar population models. It supports both photometric and spectroscopic data fitting, leveraging advanced nested sampling algorithms to explore the parameter space and estimate Bayesian evidence.

## Features

-   **Multi-Modal Fitting**: Fit photometry and spectroscopy.
-   **Robust Solvers**: Supports multiple nested sampling backends:
    -   `multinest` (via `pymultinest`) - Default, robust for multimodal posteriors. Supports parallel processing via mpirun
    -   `dynesty` (DEFAULT) - Pure Python dynamic nested sampling, supports parallel processing.
    -   `ultranest` - For high-dimensional problems (experimental support).
-   **Parallel Processing**: Accelerate `dynesty` fits using multiple CPU cores with the `--ncores` flag. For `multinest`, use `mpirun -np X mc-spf ...`.
-   **Visualization**: Includes plotting tools (`PLOT`) to visualize SEDs, Star Formation Histories (SFHs), and corner plots of posterior distributions.
-   **Modular Design**: Easily extensible for new models and filters.

## Installation

### Prerequisites

Ideally, use a dedicated Conda environment:

```bash
conda create -n mcspf python=3.13
conda activate mcspf
```

### Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/matteofox/MC-SPF.git
cd MC-SPF
python -m pip install .
```

*Note: If you encounter build isolation issues, try installing with `pip install . --no-build-isolation`.*

## Usage

The main executable script is `mc-spf`.

### Basic Command Structure

```bash
mc-spf <CATALOG> <FILTERS> <MODE> [OPTIONS]
```

-   `CATALOG`: Path to the input FITS catalog file (e.g., `TESTDATA_phot.fits`).
-   `FILTERS`: Path to the filter definition file (e.g., `filters.txt`).
-   `MODE`: Operation mode (see below).

### Operational Modes

-   `FIT`: Standard fitting mode (Simultaneous Photometry + Spectroscopy if available).
-   `FITPHOT`: Fit only Photometric data.
-   `FITSPEC`: Fit only Spectroscopic data.
-   `PLOT`: Plot results from a previous fit (requires outputs from FIT, at least`post_equal_weights.dat` and `stats.dat`).
-   `PLOTMINIMAL`: Simplified plotting mode.

### Key Options

-   `--outdir <DIR>`: Directory for output files (default: `./`).
-   `--solver <SOLVER>`: Choose nested sampling backend. Options: `dynesty` (default), `multinest`, `ultranest`.
-   `--ncores <N>`: Number of cores to use for `dynesty` solver (default: 1).
-   `--objlist <ID>`: Run only for specific object IDs (comma-separated, e.g., `1741,770`).
-   `--nlive <N>`: Number of live points for the sampler (default: 500).
-   `--sampeff <FLOAT>`: Sampling efficiency (default: 0.7).
-   `--write_models`: Write best-fit model spectra to FITS files.
    Use '-h' to see more options.

## Examples

### 1. Simple Fit with Multinest

Fit object 1741 from the test catalog using the default `dynesty` solver:

```bash
cd testdata
mc-spf TESTDATA_phot.fits filters.txt FIT --objlist 1741
```

### 2. Parallel Fit with Dynesty

Fit the same object using `dynesty` on 4 cores:

```bash
cd testdata
mc-spf TESTDATA_phot.fits filters.txt FIT --ncores 4 --objlist 1741
```

### 3. Visualizing Results

Generate summary plots for the fitted object. This mode automatically detects the solver output format:

```bash
cd testdata
mc-spf TESTDATA_phot.fits filters.txt PLOT --objlist 1741
```

## Output Structure

The code generates two main output directories:
-   `Fits/`: Contains posterior samples (`post_equal_weights.dat`), statistics (`stats.dat`), and resumed run data.
-   `Plots/`: Contains PDF summary plots (`idXXXX_summary.pdf`) and best-fit value tables.

## Authors

-   Matteo Fossati (UNIMIB) Contact for correspondence matteo.fossati@unimib.it
-   J.T. Mendel (ANU)
-   Davide Tornotti (UNIMIB)
