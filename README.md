# RANSAC Scoring Functions: Analysis and Reality Check

This project implements experiemnts of A. Shekhovtosv "RANSAC Scoring Functions: Analysis and Reality Check" [link and bib to be added].

## Overview

The project provides tools for training and evaluating learned quality measures for minimal solver hypotheses in the context of homography estimation from point correspondences.

## Table of Contents
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [PoseLib Patch](#poselib-patch)
- [Running Experiments](#running-experiments)
- [Output and Results](#output-and-results)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (requires)
- PyTorch with CUDA support
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd score_learn
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Requirements include:
   - matplotlib
   - h5py
   - opencv-python
   - kornia
   - scikit-learn
   - ipykernel


## Dataset Setup

### HEB (Homography Estimation Benchmark)

The default dataset for homography evaluation experiments.

1. **Download the HEB dataset:**
   
   Download from: https://polybox.ethz.ch/index.php/s/R5sPelZ8688It92
   Further infor at: https://github.com/danini/homography-benchmark?tab=readme-ov-file

2. **Create a symbolic link** to the dataset:
   ```bash
   # If dataset is located elsewhere
   ln -s /path/to/your/HEBHomographyDataset data/HEBHomographyDataset
   ```
   
   Or place the dataset directly:
   ```bash
   mkdir -p data
   # Copy/move dataset to data/HEBHomographyDataset/
   ```

### HPatches Dataset (Optional)

For additional experiments:

1. **Download HPatches:**
   ```bash
   cd data
   wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
   tar -xzf hpatches-sequences-release.tar.gz
   cd ..
   ```

2. **Or create symbolic link:**
   ```bash
   ln -s /path/to/hpatches-sequences-release data/hpatches-sequences-release
   ```

## PoseLib Patch

The experiments require a patched version of PoseLib with custom homography 4-point solver.

### Automatic Installation

1. **Clone PoseLib:**
   ```bash
   mkdir -p external
   cd external
   git clone https://github.com/PoseLib/PoseLib.git
   cd ../..
   ```

2. **Apply the patch using the provided script:**
   ```bash
   # Make the script executable
   chmod +x apply_poselib_patch.sh
   
   # Run the patch script
   ./apply_poselib_patch.sh
   ```
   
   The script will:
   - Verify PoseLib directory exists
   - Apply the `poselib_homography_4pt.patch`
   - Build and install the patched PoseLib
   
3. **Verify installation:**
   ```bash
   python -c "import poselib; print('homography_4pt available:', hasattr(poselib, 'homography_4pt'))"
   ```
   
   Expected output: `homography_4pt available: True`

### Manual Installation (if script fails)

1. **Apply patch manually:**
   ```bash
   cd external/PoseLib
   patch -p1 < ../../poselib_homography_4pt.patch
   ```

2. **Build and install:**
   ```bash
   mkdir -p _build
   cd _build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   cd ..
   pip install -e .
   cd ../..
   ```

## Running Experiments

### Basic Usage

Run homography evaluation on the default dataset (HEB):

```bash
python evaluate_H.py
```

### Reproducing Paper Results

For complete evaluation with variance analysis and large validation set:

```bash
python evaluate_H.py --var --largeval
```

### Command-Line Options

```bash
python evaluate_H.py [OPTIONS]
```

**Key Options:**

- `--data <dataset>`: Dataset name (default: `HEB`)
  - Available: `HEB`, `KITTI`, `PhotoTourism`, `scannet`, `eth3d`, etc.

- `--var`: Enable variance testing (multiple runs for statistical analysis)

- `--largeval`: Use large validation set for more robust results

- `--validate` / `-V`: Recompute validation metrics

- `--recompute` / `-R`: Recompute test results (ignore cached results)

- `--batch_size <N>`: Number of image pairs processed in parallel (default: 8)

- `--val_samples <N>`: Number of minimal samples for validation (default: 1000)

- `--val_pairs <N>`: Number of image pairs per scene for validation (default: 1000)

- `--val_thresholds <N>`: Grid of thresholds for validation (default: 200)

### Example Commands

**Full evaluation with variance analysis:**
```bash
python evaluate_H.py --data HEB --var --largeval
```

## Output and Results

### Results and Figures

Results and Figures are saved in the `results/<dataset>/` directory:

```
results/
├── HEB/                    # Results for HEB dataset
   └── [scene]/*.pkl              # Cached evaluation data
   └── training_size_*.pdf              # Evaluaiton plots for small random validation set series
   └── polish0/
       └── validation*.pdf              # Evaluaiton plots for large validatino set

```

## Computing Inlier Residual Distributions

The `h_patches_residuals.py` script computes and analyzes Sampson residual distributions from ground-truth inlier correspondences. This is useful for understanding the statistical properties of inliers and fitting mixture models.

### Prerequisites

Ensure the HPatches dataset is available:
```bash
ln -s /path/to/hpatches-sequences-release data/hpatches-sequences-release
```

### Running the Analysis

**Basic usage:**
```bash
python h_patches_residuals.py
```

The script will:
1. Load or process the PhotoTourism dataset residual histogram (if available)
2. Process HPatches sequences to compute ground-truth correspondences
3. Calculate Sampson residuals for both homography and essential matrix models
4. Fit chi-squared mixture models to the residual distributions
5. Generate distribution plots with fitted models

### Configuration Parameters

Key parameters can be adjusted in the script:

- `MAX_KEYPOINTS = 2000`: Maximum number of SIFT keypoints per image
- `NN_DIST_THRESHOLD = 5`: Maximum distance (pixels) to accept a match
- `DESCRIPTOR_DISTANCE_THRESHOLD = 400.0`: Maximum L2 descriptor distance
- `HIST_BINS = 100`: Number of bins for residual histograms
- `NUM_MIXTURE_COMPONENTS = 3`: Number of chi-squared components in mixture
- `INCLUDE_I_SEQUENCES = False`: Include illumination sequences (i_*)
- `INCLUDE_V_SEQUENCES = True`: Include viewpoint sequences (v_*)

### Output

**Cached results:**
- `data/hpatches-sequences-release/hpatches_symtransfer_stats.npz`: Cached residual statistics

**Generated figures:**
- `fig/PhotoTourism_dist_F.pdf`: Epipolar residual distribution from PhotoTourism data
- `fig/HPatches_dist_H.pdf`: Homography residual distribution from HPatches
- `fig/HPatches_dist_F.pdf`: Epipolar residual distribution from HPatches

**Console output includes:**
- Fitted mixture model parameters (weights and scales)
- Residual statistics (min, median, max, mean, std, percentiles)
- Number of matched correspondences per sequence

### Example Output

```
Loading descriptor distances from PhotoTourism...
Reconstructed 10000 samples from normalized histogram
Fitted 3-component Chi(1) mixture (F-residuals):
  Component 1: weight=0.6411, Chi(1, σ=0.1887)
  Component 2: weight=0.2862, Chi(1, σ=0.1887)
  Component 3: weight=0.0728, Chi(1, σ=0.1887)

Processing HPatches dataset...
Total matched correspondences: 125483
Sampson residuals summary: {'min': 0.001, 'median': 0.15, 'max': 2.98}
```

### Recomputing Results

To force recomputation (ignore cache):
```bash
# Edit the script and set:
RECOMPUTE_RES = True
```

Then run:
```bash
python h_patches_residuals.py
```

## Epipolar Geometry / Futher Experiments

For the epipolar geometry experiments, we used an external data preprocessing. It would be eaasy to replace with a proper loader of the original data and keypoint detector / matcher. The evaluation code has been changing over time and might not be fully funcitonal. It would be possible to fix and document on request.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shekhovtsov-25-RANSACA,
  title={RANSAC Scoring Functions: Analysis and Reality Check},
  author={Shekhovtsov},
  booktitle={IJCV},
  year={2025},
  note={under review}
}
```

## License

**Research Use Only License**

This software is provided for **research and educational purposes only**.

### Terms of Use

- ✅ **Permitted**: Use, modification, and distribution for academic research and educational purposes
- ✅ **Share-Alike**: Any modifications or derivative works must be distributed under the same license terms
- ✅ **Attribution**: You must cite the original work (see [Citation](#citation) section)
- ❌ **Not Permitted**: Commercial use without explicit permission

### Commercial Use

For commercial licensing or any commercial applications, please contact the author(s) at:

- Email: [author email address]
- Institution: [author institution]

### Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

### Third-Party Components

This project uses PoseLib, which has its own license terms. Please refer to the PoseLib repository for its licensing information.