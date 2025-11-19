import numpy as np

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def scale_tensor(tensor, new_min, new_max):
    scaled_tensor = tensor * (new_max - new_min) + new_min
    return scaled_tensor


import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_difference(image1, image2):
    # Read the images
    

    # Check if images are loaded properly
    if image1 is None or image2 is None:
        print("Error: One or both images not found or unable to read.")
        return

    # Ensure the images have the same dimensions
    if image1.shape != image2.shape:
        print("Error: Images do not have the same dimensions.")
        return
    
    difference = cv2.absdiff(image1, image2)
    
    normalized_difference = cv2.normalize(difference, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return normalized_difference


import matplotlib.pyplot as plt

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

def histogram_matching(reference, image):
   

    matched = match_histograms(image, reference, channel_axis=-1)
    return matched


from torch import randn
from torchmetrics.image import VisualInformationFidelity

def VIF(preds, target):
    
    vif = VisualInformationFidelity()
    
    
    return vif(preds, target)


