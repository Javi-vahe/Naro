import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .cloud_io import load_cloud
from .paths import load_case
from .replay import score
from .ros_utils import cloud_message, set_pose


class Player(Node):
    def __init__(self):
        super().__init__('scan_player')
        self.declare_parameter('case', 'courtyard')
        root, self.case = load_case(self.get_parameter('case').value)
        self.points = load_cloud(root / self.case['points'])
        self.body = np.asarray(self.case['body_odom'])
        self.cloud = self.create_publisher(PointCloud2, '/cloud_registered', qos_profile_sensor_data)
        self.odom = self.create_publisher(Odometry, '/Odometry', qos_profile_sensor_data)
        self.create_subscription(String, '/naro/result', self.on_result, 10)
        self.create_timer(.2, self.publish)

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.frame_id, odom.child_frame_id = 'camera_init', 'body'
        odom.header.stamp = stamp
        set_pose(odom.pose.pose, self.body)
        self.odom.publish(odom)
        self.cloud.publish(cloud_message(self.points, 'camera_init', stamp))

    def on_result(self, message):
        result = json.loads(message.data)
        self.get_logger().info(str(score(result, self.case)))


def main(args=None):
    rclpy.init(args=args)
    node = Player()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
