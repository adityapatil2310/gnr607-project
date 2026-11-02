import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate(det_x, det_y, det_scores, gt_centers, threshold, tolerance):
    valid = np.array(det_scores) >= threshold
    dx = np.array(det_x)[valid]
    dy = np.array(det_y)[valid]
    
    tp, fp = 0, 0
    matched_gt = set()
    
    for i in range(len(dx)):
        best_dist = float('inf')
        best_gt = -1
        
        for j, (gx, gy) in enumerate(gt_centers):
            if j in matched_gt: continue
            dist = np.sqrt((dx[i] - gx)**2 + (dy[i] - gy)**2)
            if dist <= tolerance and dist < best_dist:
                best_dist, best_gt = dist, j
                
        if best_gt != -1:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1
            
    fn = len(gt_centers) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, fn, precision, recall, f1

def plot_pr_curve(det_x, det_y, det_scores, gt_centers, tolerance, out_path):
    thresholds = np.linspace(0.3, 0.95, 20)
    precisions, recalls = [], []
    
    for th in thresholds:
        _, _, _, p, r, _ = evaluate(det_x, det_y, det_scores, gt_centers, th, tolerance)
        precisions.append(p)
        recalls.append(r)
        
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker='o', linestyle='-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()