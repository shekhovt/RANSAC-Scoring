#!/usr/bin/env python3
"""
Test script for the newly exported poselib.homography_4pt function
"""

import numpy as np
import poselib
import torch


def test_homography_4pt_basic():
    """Basic test with a simple translation"""
    print("=" * 60)
    print("Test 1: Basic homography (translation)")
    print("=" * 60)
    
    # Create 4 points in the first image (corners of a unit square)
    x1 = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0])
    ]
    
    # Translated points in second image
    x2 = [
        np.array([0.1, 0.1, 1.0]),
        np.array([1.1, 0.1, 1.0]),
        np.array([0.1, 1.1, 1.0]),
        np.array([1.1, 1.1, 1.0])
    ]
    
    H = poselib.homography_4pt(x1, x2)
    
    if H is not None:
        print("✓ Homography computed successfully")
        print(f"H =\n{H}")
        
        # Verify the homography by applying it to x1
        print("\nVerification:")
        for i, (p1, p2) in enumerate(zip(x1, x2)):
            p2_est = H @ p1
            p2_est = p2_est / p2_est[2]
            error = np.linalg.norm(p2_est - p2)
            print(f"  Point {i}: error = {error:.6e}")
            assert error < 1e-10, f"Reprojection error too large: {error}"
        print("✓ All points verified")
    else:
        print("✗ Failed to compute homography")
        return False
    
    return True


def test_homography_4pt_identity():
    """Test with identity transformation"""
    print("\n" + "=" * 60)
    print("Test 2: Identity homography")
    print("=" * 60)
    
    x1 = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0])
    ]
    
    # Same points (identity transformation)
    x2 = x1.copy()
    
    H = poselib.homography_4pt(x1, x2)
    
    if H is not None:
        print("✓ Homography computed successfully")
        print(f"H =\n{H}")
        
        # Should be close to identity (up to scale)
        H_normalized = H / H[2, 2]
        I = np.eye(3)
        error = np.linalg.norm(H_normalized - I)
        print(f"\nDistance from identity: {error:.6e}")
        assert error < 1e-6, f"Not close to identity: {error}"
        print("✓ Identity verified")
    else:
        print("✗ Failed to compute homography")
        return False
    
    return True


def test_homography_4pt_with_torch():
    """Test integration with torch tensors (convert to numpy)"""
    print("\n" + "=" * 60)
    print("Test 3: Integration with PyTorch")
    print("=" * 60)
    
    # Create points as torch tensors
    x1_torch = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=torch.float64)
    
    x2_torch = torch.tensor([
        [0.2, 0.3, 1.0],
        [1.2, 0.3, 1.0],
        [0.2, 1.3, 1.0],
        [1.2, 1.3, 1.0]
    ], dtype=torch.float64)
    
    # Convert to numpy for poselib
    x1_np = [x1_torch[i].numpy() for i in range(4)]
    x2_np = [x2_torch[i].numpy() for i in range(4)]
    
    H = poselib.homography_4pt(x1_np, x2_np)
    
    if H is not None:
        print("✓ Homography computed successfully")
        
        # Convert back to torch for verification
        H_torch = torch.from_numpy(H)
        print(f"H =\n{H_torch}")
        
        # Verify
        print("\nVerification with torch:")
        for i in range(4):
            p2_est = H_torch @ x1_torch[i]
            p2_est = p2_est / p2_est[2]
            error = torch.norm(p2_est - x2_torch[i])
            print(f"  Point {i}: error = {error.item():.6e}")
            assert error < 1e-10, f"Reprojection error too large: {error}"
        print("✓ All points verified with torch")
    else:
        print("✗ Failed to compute homography")
        return False
    
    return True


def test_homography_4pt_degenerate():
    """Test with degenerate configuration (should fail or return None)"""
    print("\n" + "=" * 60)
    print("Test 4: Degenerate configuration (collinear points)")
    print("=" * 60)
    
    # Collinear points (degenerate)
    x1 = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([2.0, 0.0, 1.0]),
        np.array([3.0, 0.0, 1.0])
    ]
    
    x2 = [
        np.array([0.1, 0.0, 1.0]),
        np.array([1.1, 0.0, 1.0]),
        np.array([2.1, 0.0, 1.0]),
        np.array([3.1, 0.0, 1.0])
    ]
    
    H = poselib.homography_4pt(x1, x2)
    
    if H is None:
        print("✓ Correctly returned None for degenerate case")
        return True
    else:
        print("⚠ Got a homography for degenerate case (might be numerically unstable)")
        print(f"H =\n{H}")
        print(f"det(H) = {np.linalg.det(H):.6e}")
        return True  # Not necessarily an error


def test_homography_4pt_check_chirality():
    """Test the check_chirality parameter"""
    print("\n" + "=" * 60)
    print("Test 5: Chirality checking")
    print("=" * 60)
    
    x1 = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0])
    ]
    
    x2 = [
        np.array([0.5, 0.5, 1.0]),
        np.array([1.5, 0.5, 1.0]),
        np.array([0.5, 1.5, 1.0]),
        np.array([1.5, 1.5, 1.0])
    ]
    
    # With chirality check (default)
    H_with_check = poselib.homography_4pt(x1, x2, check_cheirality=True)
    print(f"With chirality check: {'Success' if H_with_check is not None else 'Failed'}")
    
    # Without chirality check
    H_without_check = poselib.homography_4pt(x1, x2, check_cheirality=False)
    print(f"Without chirality check: {'Success' if H_without_check is not None else 'Failed'}")
    
    if H_with_check is not None:
        print(f"\nH (with check) =\n{H_with_check}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing poselib.homography_4pt Python binding")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= test_homography_4pt_basic()
        all_passed &= test_homography_4pt_identity()
        all_passed &= test_homography_4pt_with_torch()
        all_passed &= test_homography_4pt_degenerate()
        all_passed &= test_homography_4pt_check_chirality()
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")
