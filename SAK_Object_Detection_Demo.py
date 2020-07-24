# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
#import argparse
import imutils
import time
import cv2


#Initialize the list of labels
labels = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
bbcolor = np.random.uniform(0, 255, size=(len(labels), 3))

# load the pre-trained model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("SAK_MNetSSD_deploy.prototxt.txt","SAK_MNetSSD_deploy.caffemodel")


# initialize the video stream,
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video and set the width to 400
	fr = vs.read()
	fr = imutils.resize(fr, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = fr.shape[:2]
	rect = cv2.dnn.blobFromImage(cv2.resize(fr, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob as an input to the pre-trained network for prediction and detection
	net.setInput(rect)
	preds = net.forward()

	# loop over the detections
	for i in np.arange(0, preds.shape[2]):
		# extract the probability of the labels associated with
		# the prediction
		probs = preds[0, 0, i, 2]

		# filter out weak detections by setting the `confidence`
		# greater than 20%
		if probs > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(preds[0, 0, i, 1])
			bbox = preds[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = bbox.astype("int")

			# draw the prediction on the frame
			cat = "{}: {:.2f}%".format(labels[idx],
				probs * 100)
			cv2.rectangle(fr, (startX, startY), (endX, endY),
				bbcolor[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(fr, cat, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbcolor[idx], 2)

	# show the output frame
	cv2.imshow("Frame", fr)
	key = cv2.waitKey(1) & 0xFF

	# if the `s` key was pressed, break from the loop
	if key == ord("s"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()