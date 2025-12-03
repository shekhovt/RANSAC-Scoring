import torch
from model_H import Sampson, SampsonBM


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
    abs_error = torch.abs(sampson_bm_result - sampson_reference)
    rel_error = abs_error / (torch.abs(sampson_reference) + 1e-10)
    max_rel_error = rel_error.max().item()
    max_abs_error = abs_error.max().item()
    
    print(f"\nConsistency test results:")
    print(f"  Max relative error: {max_rel_error:.2e}")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Mean absolute error: {abs_error.mean().item():.2e}")
    
    assert torch.allclose(sampson_bm_result, sampson_reference, rtol=1e-5, atol=1e-8), \
        f"Sampson errors not consistent. Max relative error: {max_rel_error:.2e}"
    
    print("  ✓ Consistency test PASSED")


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
    
    print(f"\nEdge case test (single batch/model/point):")
    print(f"  SampsonBM: {sampson_bm.item():.6e}")
    print(f"  Sampson:   {sampson_ref.item():.6e}")
    print(f"  Difference: {abs(sampson_bm - sampson_ref).item():.2e}")
    
    assert torch.allclose(sampson_bm, sampson_ref, rtol=1e-5, atol=1e-8), \
        f"Edge case failed: {sampson_bm} vs {sampson_ref}"
    
    print("  ✓ Edge case test PASSED")


def test_sampson_consistency_identity():
    """
    Test with identity homography - errors should be zero for matching points.
    """
    B, M, n = 2, 2, 3
    
    # Identity homographies
    H_batch = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).expand(B, M, 3, 3).clone()
    
    # Same points
    x_batch = torch.randn(B, n, 2, dtype=torch.float64)
    y_batch = x_batch.clone()
    
    x_hom = torch.cat([x_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    y_hom = torch.cat([y_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    
    sampson_bm = SampsonBM(H_batch, x_hom, y_hom)
    
    max_error = sampson_bm.max().item()
    print(f"\nIdentity homography test:")
    print(f"  Max error (should be ~0): {max_error:.2e}")
    
    # Should be near zero
    assert torch.all(sampson_bm < 1e-10), \
        f"Identity homography should give near-zero errors, got max: {max_error}"
    
    print("  ✓ Identity test PASSED")


def test_sampson_large_batch():
    """
    Test with larger batches to ensure scalability.
    """
    torch.manual_seed(999)
    
    B, M, n = 5, 10, 20
    
    H_batch = torch.randn(B, M, 3, 3, dtype=torch.float64)
    x_batch = torch.randn(B, n, 2, dtype=torch.float64)
    y_batch = torch.randn(B, n, 2, dtype=torch.float64)
    
    x_hom = torch.cat([x_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    y_hom = torch.cat([y_batch, torch.ones(B, n, 1, dtype=torch.float64)], dim=-1)
    
    sampson_bm_result = SampsonBM(H_batch, x_hom, y_hom)
    
    # Sample a few random points to verify
    num_samples = 10
    max_error = 0.0
    
    print(f"\nLarge batch test (B={B}, M={M}, n={n}):")
    print(f"  Checking {num_samples} random samples...")
    
    for _ in range(num_samples):
        b = torch.randint(0, B, (1,)).item()
        m = torch.randint(0, M, (1,)).item()
        i = torch.randint(0, n, (1,)).item()
        
        sampson_bm = sampson_bm_result[b, m, i]
        sampson_ref = Sampson(H_batch[b, m], x_batch[b, i], y_batch[b, i])
        
        error = abs(sampson_bm - sampson_ref).item()
        max_error = max(max_error, error)
    
    print(f"  Max error in samples: {max_error:.2e}")
    assert max_error < 1e-6, f"Large batch test failed with error: {max_error}"
    
    print("  ✓ Large batch test PASSED")


if __name__ == "__main__":
    print("="*60)
    print("Running Sampson consistency tests")
    print("="*60)
    
    test_sampson_consistency()
    test_sampson_consistency_edge_cases()
    test_sampson_consistency_identity()
    test_sampson_large_batch()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
