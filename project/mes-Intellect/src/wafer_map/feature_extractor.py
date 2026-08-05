import numpy as np
import cv2
from scipy import stats

def extract_features(wafer_maps):
    features = []
    for wm in wafer_maps:
        feat = {}
        feat['defect_ratio'] = np.mean(wm > 0)
        feat['edge_ratio'] = np.sum(cv2.Canny((wm > 0).astype(np.uint8)*255, 50, 150) > 0) / wm.size
        contours, _ = cv2.findContours((wm > 0).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        feat['num_contours'] = len(contours)
        features.append(list(feat.values()))
    return np.array(features)