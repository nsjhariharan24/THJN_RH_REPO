#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class PitchTest(Node):
    def __init__(self):
        super().__init__('pitch_test')
        self.pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.timer = self.create_timer(0.1, self.send_cmd)

    def send_cmd(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        # No linear motion
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        # PITCH → rotation about Y axis
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.2   # rad/s
        msg.twist.angular.z = 0.0

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = PitchTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

