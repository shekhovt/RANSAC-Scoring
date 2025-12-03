# Score Learning for Homography Estimation

This project implements learned scoring functions for robust homography estimation using RANSAC-based methods.

## Overview

The project provides tools for training and evaluating learned quality measures for minimal solver hypotheses in the context of homography estimation from point correspondences.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git with submodule support

### Setup Instructions

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive <repository-url>
   cd score_learn
   ```
   
   If you already cloned without `--recursive`:
   ```bash
   git submodule update --init --recursive
   ```

2. **Create and activate a virtual environment:**
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

4. **Apply PoseLib patch and install:**
   
   This project requires a patched version of PoseLib that exports the `homography_4pt` function to Python.
   
   ```bash
   cd external/PoseLib
   
   # Apply the patch
   git apply ../../poselib_homography_4pt.patch
   
   # Build and install PoseLib
   pip install .
   
   # Return to project root
   cd ../..
   ```
   
   **Verify installation:**
   ```bash
   python -c "import poselib; print('homography_4pt available:', hasattr(poselib, 'homography_4pt'))"
   ```
   
   Expected output: `homography_4pt available: True`

5. **Alternative: Manual patch application**
   
   If the automatic patch fails, you can apply changes manually:
   
   - See `POSELIB_PATCH_README.md` for detailed instructions
   - The patch adds Python bindings for the `homography_4pt` minimal solver
   - Changes are required in `external/PoseLib/pybind/pyposelib.cc`

### Troubleshooting Installation

**Patch fails to apply:**
- Check your PoseLib version: `cd external/PoseLib && git log --oneline | head -5`
- The patch is designed for PoseLib 2.0.4
- Try: `git apply --check ../../poselib_homography_4pt.patch` to see what fails
- Follow manual instructions in `POSELIB_PATCH_README.md`

**Import error after installation:**
```bash
pip uninstall poselib
cd external/PoseLib
pip install .
```

**Function not available:**
```bash
# Verify the patch was applied
cd external/PoseLib
grep -n "homography_4pt_wrapper" pybind/pyposelib.cc
```

## Usage

### Testing the Installation

Run the provided tests:
```bash
# Test homography_4pt function
python test_homography_4pt.py

# Test Sampson error consistency
python test_sampson_H.py

# Run usage examples
python example_homography_4pt.py
```

### Training

```bash
python train.py [options]
```

### Evaluation

```bash
python evaluate_H.py [options]
```

## Project Structure

```
score_learn/
├── external/
│   ├── PoseLib/              # PoseLib submodule (patched)
│   └── homography-benchmark/ # Homography benchmark submodule
├── data/                     # Dataset directory
├── models/                   # Trained model checkpoints
├── results/                  # Evaluation results
├── poselib_homography_4pt.patch  # PoseLib patch file
├── POSELIB_PATCH_README.md  # Detailed patch documentation
├── train.py                  # Training script
├── evaluate_H.py             # Evaluation script
├── model_H.py                # Model definitions
├── sampson_H.py              # Sampson error implementations
└── requirements.txt          # Python dependencies
```

## Key Components

### PoseLib Patch

The `poselib_homography_4pt.patch` adds Python bindings for the minimal 4-point homography solver. This enables:

- Direct computation of homographies from exactly 4 point correspondences
- Integration with custom scoring functions
- Faster hypothesis generation compared to full RANSAC pipeline

**Usage example:**
```python
import poselib
import numpy as np

# 4 point correspondences in homogeneous coordinates
x1 = [np.array([0, 0, 1]), np.array([1, 0, 1]),
      np.array([0, 1, 1]), np.array([1, 1, 1])]
x2 = [np.array([0.1, 0.1, 1]), np.array([1.1, 0.1, 1]),
      np.array([0.1, 1.1, 1]), np.array([1.1, 1.1, 1])]

# Compute homography
H = poselib.homography_4pt(x1, x2, check_cheirality=True)
```

### Sampson Error

The project implements both scalar and vectorized Sampson error computation for homographies:

- `Sampson(H, x, y)` - Scalar version for single point pairs
- `SampsonBM(H, x, y)` - Batched matrix version for multiple hypotheses

See `test_sampson_H.py` for consistency verification.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{your-paper,
  title={Your Paper Title},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}
```

## Dependencies

Core dependencies:
- PyTorch
- NumPy
- PoseLib (patched version)
- OpenCV
- h5py

See `requirements.txt` for the complete list.

## License

[Your License Here]

## Contributing

The PoseLib patch can be submitted upstream:

```bash
cd external/PoseLib
git checkout -b add-homography-4pt-binding
git add pybind/pyposelib.cc
git commit -m "Add Python binding for homography_4pt solver"
# Create PR at https://github.com/vlarsson/PoseLib
```

## Contact

[Your Contact Information]
