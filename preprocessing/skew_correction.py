import numpy as np
import cv2
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

def skew_correction(gray_image):
    orig = gray_image
    # threshold to get rid of extraneous noise
    thresh = threshold_otsu(gray_image)
    normalize = gray_image > thresh
    blur = gaussian(normalize, 3)
    edges = canny(blur)
    hough_lines = probabilistic_hough_line(edges)
    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]
    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]
    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]
    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=100)
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]
    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)
    # rotate the image to deskew it
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_number, 1.0)
    rotated = cv2.warpAffine(orig, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return np.array(rotated), rotation_number