#!/usr/bin/env python3

import cv2
import numpy as np
import os
import depthai as dai
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import zmq
import base64

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

class Apple_Detection(Node):

	def __init__(self, dai_statement):
		super().__init__('apple_detection_node')
		self.get_logger().info('Start Apple Detection')

		### DepthAI ##
		self.dai_statement = dai_statement

		self.queueRgb = self.dai_statement.getOutputQueue(name='rgb', maxSize=4, blocking=False)
		self.queueDisplay = self.dai_statement.getOutputQueue(name='display', maxSize=4, blocking=False)
		self.queueNn = self.dai_statement.getOutputQueue(name='nn', maxSize=4, blocking=False)

		self.xc_list = []
		self.yc_list = []
		self.prev_slope = 0.0

		### ZMQ Streaming ###
		PORT_DISPLAY = '5555'
		context = zmq.Context()
		self.footage_socket = context.socket(zmq.PUB)
		self.footage_socket.bind('tcp://*:' + PORT_DISPLAY)
		print("Video streaming on PORT: " + PORT_DISPLAY)


		### Pub/Sub ###
		self.xc_pub = self.create_publisher(Float32, '/detection/xc', 10)
		self.xc_msg = Float32()

		timer_period = 0.02
		self.timer = self.create_timer(timer_period, self.timer_callback)


	def timer_callback(self):

		self.xc_list = []
		self.yc_list = []

		inRgb = self.queueRgb.get()
		inDisplay = self.queueDisplay.get()
		inNn = self.queueNn.get()

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

				self.xc_list.append(xc)
				self.yc_list.append(yc)

				bbox = [xmin, ymin, xmax, ymax]
				cv2.rectangle(displayFrame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
				cv2.circle(displayFrame, (xc, yc), 5, (0,50,200), -1)
				cv2.putText(displayFrame, labelMap[detection.label], ((bbox[0]+10), (bbox[1]+20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))
				cv2.putText(displayFrame, f"{int(detection.confidence * 100)}%", ((bbox[0]+10), (bbox[1]+40)), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (255,0,0))


		X = np.asarray(self.xc_list)
		Y = np.asarray(self.yc_list)

		if len(X) == 1:

			slope = 0.0
			x_ave = X[0]
			y_ave = Y[0]

			cv2.circle(displayFrame, (int(x_ave), int(y_ave)), 5, (0,255,255), -1)

		elif len(X) == 2:

			xmax_idx = np.argmax(X)
			xmin_idx = np.argmin(X)

			xmax = X[xmax_idx]
			xmin = X[xmin_idx]
			ymax = Y[xmax_idx]
			ymin = Y[xmin_idx]

			x_ave = (xmax + xmin)/2
			y_ave = (ymax + ymin)/2

			slope = (xmax - xmin) / (ymax - ymin)

			print(f"m: {slope:.2f} xmax: {xmax:.2f} ymax: {ymax:.2f} xmin: {xmin:.2f} ymin: {ymin:.2f}")
			cv2.line(displayFrame, (int(X[0]), int(Y[0])), (int(X[1]), int(Y[1])), (0, 0, 255), 2)
			cv2.circle(displayFrame, (int(x_ave), int(y_ave)), 5, (0,255,255), -1)

		elif len(X) > 2:

			tmp_X = np.copy(X)
			tmp_Y = np.copy(Y)

			line_eq = np.linalg.lstsq(np.vstack([tmp_X, np.ones(len(tmp_X))]).T, tmp_Y, rcond=None)[0]
			slope = line_eq[0]
			intercept = line_eq[1]

			xmax_idx = np.argmax(tmp_X)
			xmin_idx = np.argmin(tmp_X)

			x1 = tmp_X[xmin_idx]
			y1 = x1*slope + intercept

			x2 = tmp_X[xmax_idx]
			y2 = x2*slope + intercept

			x_ave = (x1 + x2)/2
			y_ave = (y1 + y2)/2

			print(f"m: {slope:.2f} b: {intercept:.2f} x1: {x1:.2f} y1: {y1:.2f} x2: {x2:.2f} y2: {y2:.2f}")
			cv2.line(displayFrame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
			cv2.circle(displayFrame, (int(x_ave), int(y_ave)), 5, (0,255,255), -1)

		else:
			x_ave = -1.0
			slope = self.prev_slope

		self.prev_slope = slope


		if inDisplay is not None:
			# cv2.imshow("displayFrame", displayFrame)
			# cv2.waitKey(1)

			encoded, buffer = cv2.imencode('.jpg', displayFrame)
			jpg_as_text = base64.b64encode(buffer)
			self.footage_socket.send(jpg_as_text)

		self.xc_msg.data = float(x_ave)
		self.xc_pub.publish(self.xc_msg)


def main(args=None):

	rclpy.init(args=None)

	dai_statement = dai.Device(pipeline)

	node = Apple_Detection(dai_statement)
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == "__main__":
	main()