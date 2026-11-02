import numpy as np

def ncc_2d(image, template, mask=None, chunk_size=100):
    """
    Computes zero-mean Normalized Cross-Correlation.
    Uses chunked sliding windows to prevent memory explosions on large images.
    """
    H, W = image.shape
    h, w = template.shape
    
    if mask is None:
        mask = np.ones((h, w), dtype=np.float32)
        
    out_H, out_W = H - h + 1, W - w + 1
    ncc_map = np.zeros((out_H, out_W), dtype=np.float32)
    
    mask_sum = np.sum(mask)
    if mask_sum == 0: return ncc_map
    
    # Zero-mean template (only over masked region)
    t_mean = np.sum(template * mask) / mask_sum
    t_prime = (template - t_mean) * mask
    t_var = np.sum(t_prime**2)
    
    # Process image in row chunks
    for i in range(0, out_H, chunk_size):
        end_i = min(i + chunk_size, out_H)
        chunk = image[i : end_i + h - 1, :]
        
        # Extract sliding windows: shape (rows, cols, h, w)
        windows = np.lib.stride_tricks.sliding_window_view(chunk, (h, w))
        
        # Local sums and sums of squares (masked)
        sum_f = np.tensordot(windows, mask, axes=([-2, -1], [0, 1]))
        sum_f2 = np.tensordot(windows**2, mask, axes=([-2, -1], [0, 1]))
        
        var_f = sum_f2 - (sum_f**2) / mask_sum
        var_f = np.maximum(var_f, 0) # Float precision safety
        
        # Cross correlation
        cross = np.tensordot(windows, t_prime, axes=([-2, -1], [0, 1]))
        
        denom = np.sqrt(var_f * t_var)
        
        # Epsilon/Variance check: avoid division by zero in flat regions
        valid = denom > 1e-8
        
        ncc_chunk = np.zeros_like(cross)
        ncc_chunk[valid] = cross[valid] / denom[valid]
        ncc_chunk[~valid] = 0.0
        
        ncc_map[i:end_i, :] = ncc_chunk
        
    return ncc_map

def validate_ncc():
    # Hidden validation using a synthetic patch
    img = np.random.rand(100, 100).astype(np.float32)
    temp = img[40:60, 40:60].copy()
    
    # Exact match validation
    res1 = ncc_2d(img, temp)
    peak1 = np.max(res1)
    
    # Brightness/Contrast change validation (a * f + b)
    temp_changed = temp * 1.5 + 50.0
    res2 = ncc_2d(img, temp_changed)
    peak2 = np.max(res2)
    
    print(f"Validation Exact Peak (Expected ~1.0): {peak1:.5f}")
    print(f"Validation Transform Peak (Expected ~1.0): {peak2:.5f}")

if __name__ == "__main__":
    validate_ncc()