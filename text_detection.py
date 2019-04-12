from imutils.object_detection import non_max_suppression
from utils import decode_predictions
import numpy as np
import time
import cv2 as cv


def detect_text(image_path,net_path,filter=False,padding=0.05):
	# image_path = "rctest/test1.jpg"

	img = cv.imread(image_path)
	o_img = img.copy()
	#width and height must be a multiple of 32 for EAST detection
	o_h,o_w = o_img.shape[:2]
	new_h,new_w = 256 , 256

	rh =  img.shape[0]  / float(new_h)
	rw =  img.shape[1]  / float(new_w)

	img = cv.resize(img,(new_h,new_w))

	if filter:
		img = cv.bilateralFilter(img,9,75,75)

	# the first layer outputs probabilities and the
	# second layer can be used to derive the bounding box 
	# coordinates of text
	layerNames = ["feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

	net = cv.dnn.readNet(net_path)

	blob = cv.dnn.blobFromImage(img,1.0,img.shape[:2][::-1],
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	# perform a forward pass
	# of the model to obtain the two output layer sets
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	

	predictions = []

	for (startX, startY, endX, endY) in boxes:
		
		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)

		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(o_w, endX + (dX * 2))
		endY = min(o_h, endY + (dY * 2))
		
		predictions.append([startX,startY,endX,endY])
	


	return img,predictions


