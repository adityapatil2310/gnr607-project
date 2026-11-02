import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from ncc import ncc_2d
from detect import detect_peaks, plot_detections
from rotate_scale import rotate_image, scale_image
from evaluate import evaluate

def run_all(scene, gt, template, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    
    OP_THRESH = 0.55
    NMS_RADIUS = 30
    TOLERANCE = 30
    results_csv = open(os.path.join(base_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(results_csv)
    writer.writerow(['Experiment', 'Angle/Scale', 'Threshold', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Peak NCC'])

    print("Running E1: Baseline...", flush=True)
    ncc_map = ncc_2d(scene, template)
    dx, dy, ds = detect_peaks(ncc_map, 0.3, NMS_RADIUS, template.shape)
    plot_detections(scene, dx, dy, ncc_map, os.path.join(base_dir, 'E1_baseline.png'))
    tp, fp, fn, p, r, f1 = evaluate(dx, dy, ds, gt, OP_THRESH, TOLERANCE)
    peak = np.max(ncc_map)
    writer.writerow(['E1_Baseline', 'None', OP_THRESH, tp, fp, fn, p, r, f1, peak])
    print(f"E1: complete (recall={r:.3f}, peak NCC={peak:.3f})", flush=True)

    print("Running E2: Rotation")
    angles = [0, 15, 30, 45, 90, 135, 180]
    recalls, peaks = [], []
    for index, ang in enumerate(angles, start=1):
        print(f"E2 Rotation [{index}/{len(angles)}]: angle={ang} degrees...", flush=True)
        rot_t, mask_t = rotate_image(template, ang)
        n_map = ncc_2d(scene, rot_t, mask=mask_t)
        rx, ry, rs = detect_peaks(n_map, 0.3, NMS_RADIUS, rot_t.shape)
        tp, fp, fn, p, rec, f1 = evaluate(rx, ry, rs, gt, OP_THRESH, TOLERANCE)
        pk = np.max(n_map)
        recalls.append(rec)
        peaks.append(pk)
        writer.writerow(['E2_Rotate', ang, OP_THRESH, tp, fp, fn, p, rec, f1, pk])
        print(f"E2 Rotation [{index}/{len(angles)}]: complete (recall={rec:.3f}, peak NCC={pk:.3f})", flush=True)
        
    plt.figure()
    plt.plot(angles, recalls, label="Recall")
    plt.plot(angles, peaks, label="Peak Score")
    plt.legend()
    plt.title("Rotation vs Performance")
    plt.savefig(os.path.join(base_dir, 'E2_rotation_plot.png'))
    plt.close()

    print("Running E3: Scale")
    scales = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]
    recalls, peaks = [], []
    for index, sc in enumerate(scales, start=1):
        print(f"E3 Scale [{index}/{len(scales)}]: scale={sc}...", flush=True)
        sc_t, mask_t = scale_image(template, sc)
        s_map = ncc_2d(scene, sc_t, mask=mask_t)
        sx, sy, ss = detect_peaks(s_map, 0.3, NMS_RADIUS, sc_t.shape)
        tp, fp, fn, p, rec, f1 = evaluate(sx, sy, ss, gt, OP_THRESH, TOLERANCE)
        pk = np.max(s_map)
        recalls.append(rec)
        peaks.append(pk)
        writer.writerow(['E3_Scale', sc, OP_THRESH, tp, fp, fn, p, rec, f1, pk])
        print(f"E3 Scale [{index}/{len(scales)}]: complete (recall={rec:.3f}, peak NCC={pk:.3f})", flush=True)

    print("Running E4: Mitigation (Template Bank)")
    # Using max across a small bank: rotations 0, 45, 90
    mitigation_map = np.zeros_like(scene)
    bank_angles = [0, 45, 90]
    for index, ang in enumerate(bank_angles, start=1):
        print(f"E4 Template Bank [{index}/{len(bank_angles)}]: rotation={ang} degrees...", flush=True)
        t_bank, m_bank = rotate_image(template, ang)
        m_map = ncc_2d(scene, t_bank, mask=m_bank)
        pad_h = mitigation_map.shape[0] - m_map.shape[0]
        pad_w = mitigation_map.shape[1] - m_map.shape[1]
        m_map_padded = np.pad(m_map, ((0, pad_h), (0, pad_w)), mode='constant')
        mitigation_map = np.maximum(mitigation_map, m_map_padded)
        print(f"E4 Template Bank [{index}/{len(bank_angles)}]: complete", flush=True)
        
    mx, my, ms = detect_peaks(mitigation_map, 0.3, NMS_RADIUS, template.shape)
    tp, fp, fn, p, r, f1 = evaluate(mx, my, ms, gt, OP_THRESH, TOLERANCE)
    writer.writerow(['E4_Mitigation', 'Bank_0_45_90', OP_THRESH, tp, fp, fn, p, r, f1, np.max(mitigation_map)])
    
    results_csv.close()
    print("Experiments Complete. Metrics saved.")