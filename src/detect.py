import numpy as np
import matplotlib.pyplot as plt
import os

def detect_peaks(ncc_map, threshold, radius, template_shape):
    th, tw = template_shape
    ys, xs = np.where(ncc_map >= threshold)
    scores = ncc_map[ys, xs]
    
    # Adjust coordinates to point to the center of the template
    xs = xs + tw // 2
    ys = ys + th // 2
    
    keep_x, keep_y, keep_scores = [], [], []
    indices = np.argsort(scores)[::-1]
    
    while len(indices) > 0:
        current = indices[0]
        keep_x.append(xs[current])
        keep_y.append(ys[current])
        keep_scores.append(scores[current])
        
        if len(indices) == 1: break
            
        dist_sq = (xs[indices[1:]] - xs[current])**2 + (ys[indices[1:]] - ys[current])**2
        keep_idx = np.where(dist_sq > radius**2)[0]
        indices = indices[1:][keep_idx]
        
    return keep_x, keep_y, keep_scores

def plot_detections(image, det_x, det_y, ncc_map, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image, cmap='gray')
    ax1.scatter(det_x, det_y, edgecolor='red', facecolor='none', s=80, lw=2)
    ax1.set_title("Detections")
    ax1.axis('off')
    
    im2 = ax2.imshow(ncc_map, cmap='viridis', vmin=-1, vmax=1)
    ax2.set_title("NCC Heatmap")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()