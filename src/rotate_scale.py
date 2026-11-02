import numpy as np

def bilinear_interpolate(image, x, y):
    h, w = image.shape
    x0 = np.clip(np.floor(x).astype(int), 0, w-1)
    x1 = np.clip(x0 + 1, 0, w-1)
    y0 = np.clip(np.floor(y).astype(int), 0, h-1)
    y1 = np.clip(y0 + 1, 0, h-1)

    Ia, Ib, Ic, Id = image[y0, x0], image[y1, x0], image[y0, x1], image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def rotate_image(image, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    h, w = image.shape
    
    corners = np.array([[0,0], [w,0], [0,h], [w,h]])
    new_corners = corners @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)
    new_w, new_h = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
    
    X, Y = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    # Inverse map
    X_src = (X + min_x) * cos_a - (Y + min_y) * sin_a
    Y_src = (X + min_x) * sin_a + (Y + min_y) * cos_a
    
    valid = (X_src >= 0) & (X_src <= w-1) & (Y_src >= 0) & (Y_src <= h-1)
    
    out_img = np.zeros((new_h, new_w), dtype=np.float32)
    out_mask = np.zeros((new_h, new_w), dtype=np.float32)
    
    out_img[valid] = bilinear_interpolate(image, X_src[valid], Y_src[valid])
    out_mask[valid] = 1.0
    return out_img, out_mask

def scale_image(image, scale):
    h, w = image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    X, Y = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    X_src, Y_src = X / scale, Y / scale
    out_img = bilinear_interpolate(image, X_src, Y_src)
    out_mask = np.ones((new_h, new_w), dtype=np.float32)
    
    return out_img, out_mask