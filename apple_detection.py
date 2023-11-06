#!/usr/bin/env python3

import cv2
import numpy as np
import os
import depthai as dai
import time

labelMap = [
	"person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
	"truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
	"bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
	"bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
	"suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
	"baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
	"fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
	"orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
	"chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
	"laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
	"toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
	"teddy bear",     "hair drier", "toothbrush"
]

## Create Pipeline ##
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
detection_nn = pipeline.createYoloDetectionNetwork()


## RGB camera ##
camRgb.setPreviewKeepAspectRatio(False)
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)
camRgb.setIspScale(2,3)

## Yolo Detection NN ##
project_dir = os.getcwd()
blob_dir = "models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob"
detection_nn.setBlobPath(os.path.join(project_dir, blob_dir))
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detection_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detection_nn.setIouThreshold(0.5)
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)

## Xout ##
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNn = pipeline.create(dai.node.XLinkOut)
xoutDisplay = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNn.setStreamName("nn")
xoutDisplay.setStreamName("display")

camRgb.preview.link(detection_nn.input)
detection_nn.passthrough.link(xoutRgb.input)
camRgb.video.link(xoutDisplay.input)
detection_nn.out.link(xoutNn.input)

## DAI statement ##

dai_statement = dai.Device(pipeline)

queueRgb = dai_statement.getOutputQueue(name='rgb', maxSize=4, blocking=False)
queueDisplay = dai_statement.getOutputQueue(name='display', maxSize=4, blocking=False)
queueNn = dai_statement.getOutputQueue(name='nn', maxSize=4, blocking=False)

while True:

	inRgb = queueRgb.get()
	inDisplay = queueDisplay.get()
	inNn = queueNn.get()

	if inRgb is not None:
		frame = inRgb.getCvFrame()

	if inDisplay is not None:
		displayFrame = inDisplay.getCvFrame()

	if inNn is not None:
		detections = inNn.detections


	for detection in detections:

		if detection.label == 47:

			height = displayFrame.shape[0]
			width = displayFrame.shape[1]

			if np.isinf(detection.xmin) or np.isinf(detection.ymin) or np.isinf(detection.xmax) or np.isinf(detection.ymax):
				print("There is infinity")
				print(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
				continue


			xmin = int(detection.xmin*width)
			ymin = int(detection.ymin*height)
			xmax = int(detection.xmax*width)
			ymax = int(detection.ymax*height)

			xc = int((xmax+xmin)/2.0)
			yc = int((ymax+ymin)/2.0)

			bbox = [xmin, ymin, xmax, ymax]
			cv2.rectangle(displayFrame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
			cv2.circle(displayFrame, (xc, yc), 5, (0,50,200), -1)
			cv2.putText(displayFrame, labelMap[detection.label], ((bbox[0]+10), (bbox[1]+20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))
			cv2.putText(displayFrame, f"{int(detection.confidence * 100)}%", ((bbox[0]+10), (bbox[1]+40)), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (255,0,0))

	if inDisplay is not None:
		cv2.imshow("displayFrame", displayFrame)
		cv2.waitKey(1)