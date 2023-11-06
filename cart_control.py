#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class CartControl(Node):

	def __init__(self):
		super().__init__('cart_control_node')
		self.get_logger().info('Start Cart Control')

		## Cart parameters ##
		self.const_vx = 0.3 # 30cm/s
		self.wz_steering = 2.0
		self.prev_vx = 0.0

		## Detection variables ##
		self.xc = -1.0
		self.frame_width = 1280
		self.frame_width_half = self.frame_width/2
		self.frame_db = 40
		self.frame_min_db = (self.frame_width_half - self.frame_db)
		self.frame_max_db = (self.frame_width_half + self.frame_db)


		## Pub/Sub ##
		self.xc_sub = self.create_subscription(Float32, '/detection/xc', self.xc_callback, 10)
		self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.cmd_vel_msg = Twist()

		timer_period = 0.05 # 20Hz
		self.timer = self.create_timer(timer_period, self.timer_callback)


	def xc_callback(self, msg):

		self.xc = msg.data 

	def map_with_limit(self, val, in_min, in_max, out_min, out_max):

		m = (out_max - out_min)/(in_max - in_min)
		out = m*(val - in_min) + out_min

		if out_min > out_max:
			if out > out_min:
				out = out_min
			elif out < out_max:
				out = out_max
			else:
				pass
		elif out_max > out_min:
			if out > out_max:
				out = out_max
			elif out < out_min:
				out = out_min
			else:
				pass
		else:
			pass

		return out

	def timer_callback(self):

		if self.xc == -1.0:
			vx = 0.0
			wz = 0.0

			drive_state = 'STOP'

		elif self.frame_min_db <= self.xc <= self.frame_max_db:

			vx = self.const_vx
			wz = 0.0

			drive_state = '_GO_'

		elif self.xc < self.frame_min_db:

			vx = self.const_vx
			wz = self.map_with_limit(self.xc, 0, self.frame_min_db, self.wz_steering, 0.0)

			drive_state = 'LEFT'

		elif self.frame_max_db < self.xc:

			vx = self.const_vx
			wz = self.map_with_limit(self.xc, self.frame_max_db, self.frame_width, 0.0, -self.wz_steering)

			drive_state = "RIGH"

		else:
			vx = 0.0
			wz = 0.0

			drive_state = "_WTF"


		vx_smooth = self.prev_vx*0.9 + vx*0.1
		vx = vx_smooth
		self.prev_vx = vx_smooth

		self.cmd_vel_msg.linear.x = vx
		self.cmd_vel_msg.angular.z = wz
		self.cmd_vel_pub.publish(self.cmd_vel_msg)

		print("{} | xc: {:.2f} vx: {:.2f} wz: {:.2f}".format(\
			drive_state, self.xc, vx, wz))

def main(args=None):

	rclpy.init(args=None)
	node = CartControl()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == "__main__":
	main()