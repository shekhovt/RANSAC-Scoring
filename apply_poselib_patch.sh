#!/bin/bash
# Script to apply the homography_4pt patch to PoseLib
# Usage: ./apply_poselib_patch.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSELIB_DIR="${SCRIPT_DIR}/external/PoseLib"
PATCH_FILE="${SCRIPT_DIR}/poselib_homography_4pt.patch"

echo "========================================"
echo "PoseLib homography_4pt Patch Installer"
echo "========================================"
echo ""

# Check if PoseLib directory exists
if [ ! -d "$POSELIB_DIR" ]; then
    echo "ERROR: PoseLib directory not found at: $POSELIB_DIR"
    echo "Please ensure PoseLib is cloned to external/PoseLib"
    exit 1
fi

# Check if patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found at: $PATCH_FILE"
    exit 1
fi

cd "$POSELIB_DIR"

# Check if patch is already applied
if grep -q "homography_4pt_wrapper" pybind/pyposelib.cc; then
    echo "✓ Patch appears to be already applied"
    echo ""
    echo "To verify, check if poselib.homography_4pt() is available:"
    echo "  python -c 'import poselib; print(hasattr(poselib, \"homography_4pt\"))'"
    echo ""
    read -p "Do you want to reinstall PoseLib anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation."
        exit 0
    fi
else
    # Try to apply the patch
    echo "Applying patch to PoseLib..."
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        echo "✓ Patch can be applied cleanly"
        git apply "$PATCH_FILE"
        echo "✓ Patch applied successfully"
    elif patch -p1 --dry-run < "$PATCH_FILE" >/dev/null 2>&1; then
        echo "✓ Patch can be applied with patch command"
        patch -p1 < "$PATCH_FILE"
        echo "✓ Patch applied successfully"
    else
        echo "ERROR: Patch cannot be applied automatically."
        echo "This may be due to:"
        echo "  1. PoseLib version mismatch"
        echo "  2. Local modifications to pyposelib.cc"
        echo ""
        echo "Please apply the changes manually by editing:"
        echo "  $POSELIB_DIR/pybind/pyposelib.cc"
        echo ""
        echo "See POSELIB_PATCH_README.md for manual instructions."
        exit 1
    fi
fi

# Build and install PoseLib
echo ""
echo "Building and installing PoseLib..."
echo "This may take a few minutes..."
echo ""

# Uninstall existing poselib
pip uninstall -y poselib 2>/dev/null || true

# Install new version
if pip install . --verbose 2>&1 | tail -20; then
    echo ""
    echo "✓ PoseLib installed successfully"
else
    echo ""
    echo "ERROR: Failed to install PoseLib"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."
if python -c "import poselib; assert hasattr(poselib, 'homography_4pt'), 'homography_4pt not found'" 2>/dev/null; then
    echo "✓ homography_4pt function is available"
    echo ""
    echo "========================================"
    echo "Installation successful!"
    echo "========================================"
    echo ""
    echo "You can now use: poselib.homography_4pt()"
    echo ""
    echo "Example:"
    echo "  import poselib, numpy as np"
    echo "  x1 = [np.array([0, 0, 1]), np.array([1, 0, 1]),"
    echo "        np.array([0, 1, 1]), np.array([1, 1, 1])]"
    echo "  x2 = [np.array([0.1, 0.1, 1]), np.array([1.1, 0.1, 1]),"
    echo "        np.array([0.1, 1.1, 1]), np.array([1.1, 1.1, 1])]"
    echo "  H = poselib.homography_4pt(x1, x2)"
    echo ""
    echo "Run tests:"
    echo "  python test_homography_4pt.py"
    echo "  python example_homography_4pt.py"
else
    echo "WARNING: homography_4pt function not found in poselib"
    echo "Installation may have failed. Please check the output above."
    exit 1
fi
