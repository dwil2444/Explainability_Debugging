from skimage.metrics import structural_similarity
import cv2
import numpy as np
import argparse
import os
SSIM_RES_FILE = 'dump/ssim.txt'

def compute_rise_ssim(uncal_img, cal_img):
    """
    """
    # cal = cv2.imread(args.cal)
    # uncal = cv2.imread(args.unc)
    (score, diff) = structural_similarity(cv2.imread(uncal_img), cv2.imread(cal_img), full=True, multichannel=True)
    with open(SSIM_RES_FILE, 'a') as f:
        f.writelines(str(round(score, 3)) + '\n')
    f.close()
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training params')
    parser.add_argument('--unc', type=str, default='/u/dw3zn/Repos/saliency_calibration/rise_uncalibrated.png',
                        help='Uncalibrated saliency map')
    parser.add_argument('--cal', type=str, default='/u/dw3zn/Repos/saliency_calibration/rise_calibrated.png',
    help='Calibrated saliency map')
    args = parser.parse_args()
    compute_rise_ssim()