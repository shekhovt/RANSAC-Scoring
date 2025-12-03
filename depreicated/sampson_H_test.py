import os,sys
if __name__ == "__main__":   
    __name__ = 'score_learn.tests.sampson_H_test'
    __package__ = 'score_learn.tests'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False


import torch
# import pytest
from ..model_H import Sampson, SampsonBM


def test_sampson_consistency():
    """
    Verify that SampsonBM produces the same results as the non-vectorized Sampson
    for various batch sizes, number of models, and points.
    """
    torch.manual_seed(42)
    
    # Test parameters
    B = 2  # batch size
    M = 3  # number of model hypotheses
    n = 5  # number of points
    
    # Generate random homographies [B, M, 3, 3]
    H_batch = torch.randn(B, M, 3, 3, dtype=torch.float64)
    
    # Generate random points [B, n, 2]
    x_batch = torch.randn(B, n, 2, dtype=torch.float64)
    y_batch = torch.randn(B, n, 2, dtype=torch.float64)
    
    # Convert to homogeneous coordinates for SampsonBM
    x_hom = torch.cat([x_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    y_hom = torch.cat([y_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    
    # Compute vectorized Sampson errors
    sampson_bm_result = SampsonBM(H_batch, x_hom, y_hom)  # [B, M, n]
    
    # Compute non-vectorized Sampson errors for comparison
    sampson_reference = torch.zeros(B, M, n, dtype=torch.float64)
    
    for b in range(B):
        for m in range(M):
            for i in range(n):
                H = H_batch[b, m]
                x = x_batch[b, i]
                y = y_batch[b, i]
                sampson_reference[b, m, i] = Sampson(H, x, y)
    
    # Check consistency
    assert sampson_bm_result.shape == sampson_reference.shape, \
        f"Shape mismatch: {sampson_bm_result.shape} vs {sampson_reference.shape}"
    
    # Allow for small numerical differences
    rel_error = torch.abs(sampson_bm_result - sampson_reference) / (torch.abs(sampson_reference) + 1e-10)
    max_rel_error = rel_error.max().item()
    
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Max absolute error: {torch.abs(sampson_bm_result - sampson_reference).max().item():.2e}")
    
    assert torch.allclose(sampson_bm_result, sampson_reference, rtol=1e-5, atol=1e-8), \
        f"Sampson errors not consistent. Max relative error: {max_rel_error:.2e}"


def test_sampson_consistency_edge_cases():
    """
    Test edge cases: single batch, single model, single point.
    """
    torch.manual_seed(123)
    
    # Single everything
    H_batch = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    x_batch = torch.randn(1, 1, 2, dtype=torch.float64)
    y_batch = torch.randn(1, 1, 2, dtype=torch.float64)
    
    x_hom = torch.cat([x_batch, torch.ones(1, 1, 1, dtype=torch.float64)], dim=-1)
    y_hom = torch.cat([y_batch, torch.ones(1, 1, 1, dtype=torch.float64)], dim=-1)
    
    sampson_bm = SampsonBM(H_batch, x_hom, y_hom)[0, 0, 0]
    sampson_ref = Sampson(H_batch[0, 0], x_batch[0, 0], y_batch[0, 0])
    
    assert torch.allclose(sampson_bm, sampson_ref, rtol=1e-5, atol=1e-8), \
        f"Edge case failed: {sampson_bm} vs {sampson_ref}"


def test_sampson_consistency_identity():
    """
    Test with identity homography - errors should be zero for matching points.
    """
    B, M, n = 2, 2, 3
    
    # Identity homographies
    H_batch = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).expand(B, M, 3, 3)
    
    # Same points
    x_batch = torch.randn(B, n, 2, dtype=torch.float64)
    y_batch = x_batch.clone()
    
    x_hom = torch.cat([x_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    y_hom = torch.cat([y_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    
    sampson_bm = SampsonBM(H_batch, x_hom, y_hom)
    
    # Should be near zero
    assert torch.all(sampson_bm < 1e-10), \
        f"Identity homography should give near-zero errors, got max: {sampson_bm.max()}"


if __run__:
    test_sampson_consistency()
    test_sampson_consistency_edge_cases()
    test_sampson_consistency_identity()
    print("All tests passed!")