# Score Learning for Homography Estimation

This project implements learned scoring functions for robust homography estimation using RANSAC-based methods.

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

1. **Download the HEB dataset** following the instructions from the dataset provider.

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

## Epipolar Geometry / Futher Experiments

For the epipolar geometry experiments, we used an external data preprocessing. It would be eaasy to replace with a proper loader of the original data and keypoint detector / matcher. The evaluation code has been changing over time and might not be fully funcitonal. It would be possible to fix and document on request.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shekhovtsov-25-RANSACA,
  title={RANSAC Scoring Functions: \\ Analysis and Reality Check},
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