
from text_detection import detect_text
import pytesseract
import cv2 as cv

net_path = "frozen_east_text_detection.pb"
image_path = "rctest/example_05.jpg"

img,boxes = detect_text(image_path,net_path)

config = ("-l eng --oem 1 --psm 7")

def read_text(boxes,config):
	results =[]

	for (startX, startY, endX, endY) in boxes:
		
		cv.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        roi = img[startY:endY, startX:endX]
        text = pytesseract.image_to_string(roi, config=config)
		
		results.append(((startX,startY,endX,endY),text))
	return results

predicted_text = read_text(boxes,config)

for ((startX, startY, endX, endY), text) in predicted_text:
 
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = img.copy()
	cv.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv.putText(output, text, (startX, startY - 5),
		cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
 
	# show the output image
	cv.imshow("Text Detection", output)
	cv.waitKey(0)