import cv2
from skew_correction import skew_correction

image = cv2.imread("3.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))

image = clahe.apply(image)

_, image = cv2.threshold(image, thresh=165, maxval=255, type=cv2.THRESH_TRUNC+cv2.THRESH_OTSU)

image, rotation_number = skew_correction(image)

# DIDN'T WORK :(
# ratio = 640.0 / image.shape[1]
# image.resize((int(image.shape[0] * ratio), 640, 1))
# image.resize((500, 800))

image = cv2.copyMakeBorder(
	src=image,
	top=20,
	bottom=20,
	left=20,
	right=20,
	borderType=cv2.BORDER_CONSTANT,
	value=(255, 255, 255))

cv2.imshow("gfsdg", image)
cv2.waitKey(0)