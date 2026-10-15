from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from .api import localize
from .cloud_io import load_cloud
from .ros_utils import cloud_message, read_xyz, rotation, set_pose


class Relocalizer(Node):
    def __init__(self):
        super().__init__('naro')
        defaults = dict(map_dir='', cloud_topic='/cloud_registered', odom_topic='/Odometry',
                        map_frame='map', odom_frame='camera_init', base_frame='body',
                        collect_seconds=3., max_points=150000, min_points=2000,
                        max_translation=.15, max_rotation_deg=2., sync_tolerance=.2,
                        lidar_translation=[0., 0., 0.], threads=4, auto_start=True, publish_tf=True)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.cfg = {key: self.get_parameter(key).value for key in defaults}
        self.map_dir = Path(self.cfg['map_dir']).expanduser()
        if not (self.map_dir / 'map.npz').is_file() or not (self.map_dir / 'localization_cloud.npy').is_file():
            raise ValueError('map_dir must contain map.npz and localization_cloud.npy')
        if self.cfg['map_frame'] == self.cfg['odom_frame']:
            raise ValueError('map_frame and odom_frame must differ')
        if min(self.cfg['collect_seconds'], self.cfg['max_points'], self.cfg['min_points'], self.cfg['threads']) <= 0:
            raise ValueError('collection duration, point limits and threads must be positive')
        lever = np.asarray(self.cfg['lidar_translation'], dtype=float)
        if lever.shape != (3,) or not np.isfinite(lever).all():
            raise ValueError('lidar_translation must contain three finite values')
        self.frame_warnings = set()
        self.odom = deque(maxlen=300)
        self.scans = deque()
        self.start_pose = None
        self.start_stamp = None
        self.point_count = 0
        self.armed = bool(self.cfg['auto_start'])
        self.future = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.tf = StaticTransformBroadcaster(self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.result = self.create_publisher(String, '~/result', latched)
        self.transform = self.create_publisher(TransformStamped, '~/transform', latched)
        self.pose = self.create_publisher(PoseStamped, '~/pose', latched)
        self.map_pub = self.create_publisher(PointCloud2, '~/map', latched)
        self.scan_pub = self.create_publisher(PointCloud2, '~/aligned', latched)
        self.create_subscription(Odometry, self.cfg['odom_topic'], self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.cfg['cloud_topic'], self.on_cloud, qos_profile_sensor_data)
        self.create_service(Trigger, '~/relocalize', self.request)
        self.create_timer(.1, self.poll)
        points = load_cloud(self.map_dir / 'localization_cloud.npy')
        points = points[::max(1, int(np.ceil(len(points) / 200000)))]
        self.map_pub.publish(cloud_message(points, self.cfg['map_frame'], self.get_clock().now().to_msg()))
        self.get_logger().info('Waiting for stationary point clouds and odometry')

    def reset_scan(self):
        self.scans.clear()
        self.point_count = 0
        self.start_stamp = None
        self.start_pose = None

    def request(self, request, response):
        if self.future is not None:
            response.success = False
            response.message = 'localization is running'
        else:
            self.reset_scan()
            self.armed = True
            response.success = True
            response.message = 'collecting a new scan'
        return response

    @staticmethod
    def stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def on_odom(self, message):
        if message.header.frame_id != self.cfg['odom_frame'] or message.child_frame_id != self.cfg['base_frame']:
            if 'odom' not in self.frame_warnings:
                self.get_logger().warning('Odometry frames do not match odom_frame and base_frame')
                self.frame_warnings.add('odom')
            return
        try:
            matrix = np.eye(4)
            matrix[:3, :3] = rotation(message.pose.pose.orientation)
            p = message.pose.pose.position
            matrix[:3, 3] = [p.x, p.y, p.z]
            if np.isfinite(matrix).all():
                self.odom.append((self.stamp(message), matrix))
        except ValueError:
            return

    def on_cloud(self, message):
        if not self.armed or self.future is not None or not self.odom:
            return
        if message.header.frame_id != self.cfg['odom_frame']:
            if 'cloud' not in self.frame_warnings:
                self.get_logger().warning('Point cloud frame does not match odom_frame')
                self.frame_warnings.add('cloud')
            return
        stamp = self.stamp(message)
        odom_stamp, body = min(self.odom, key=lambda item: abs(item[0] - stamp))
        if abs(odom_stamp - stamp) > self.cfg['sync_tolerance']:
            return
        points = read_xyz(message)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        if self.start_stamp is not None and stamp <= self.start_stamp:
            self.reset_scan()
        if self.start_pose is not None:
            shift = np.linalg.norm(body[:3, 3] - self.start_pose[:3, 3])
            delta = self.start_pose[:3, :3].T @ body[:3, :3]
            angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1.) / 2., -1., 1.)))
            if shift > self.cfg['max_translation'] or angle > self.cfg['max_rotation_deg']:
                self.reset_scan()
        if self.start_stamp is None:
            self.start_stamp, self.start_pose = stamp, body.copy()
        limit = self.cfg['max_points']
        points = points[::max(1, int(np.ceil(len(points) / limit)))]
        self.scans.append(points)
        self.point_count += len(points)
        while self.point_count > limit and len(self.scans) > 1:
            self.point_count -= len(self.scans.popleft())
        if stamp - self.start_stamp < self.cfg['collect_seconds'] or self.point_count < self.cfg['min_points']:
            return
        query = np.concatenate(list(self.scans))
        here = body[:3, 3].copy()
        sensor = here + body[:3, :3] @ np.asarray(self.cfg['lidar_translation'], dtype=float)
        self.pending = query, body.copy(), message.header.stamp
        self.armed = False
        self.future = self.pool.submit(localize, query, self.map_dir, here, sensor, self.cfg['threads'])
        self.reset_scan()
        self.get_logger().info('Relocalizing')

    def poll(self):
        if self.future is None or not self.future.done():
            return
        try:
            reply = self.future.result()
        except Exception as error:
            reply = dict(accepted=False, reason=str(error))
        self.future = None
        self.result.publish(String(data=json.dumps(reply, allow_nan=False)))
        self.get_logger().info(reply.get('reason', 'completed'))
        if not reply.get('accepted'):
            return
        query, body, stamp = self.pending
        matrix = np.asarray(reply['T_map_odom'])
        transform = TransformStamped()
        transform.header.frame_id = self.cfg['map_frame']
        transform.header.stamp = stamp
        transform.child_frame_id = self.cfg['odom_frame']
        set_pose(transform.transform, matrix)
        self.transform.publish(transform)
        if self.cfg['publish_tf']:
            self.tf.sendTransform(transform)
        pose = PoseStamped()
        pose.header.frame_id = self.cfg['map_frame']
        pose.header.stamp = stamp
        set_pose(pose.pose, matrix @ body)
        self.pose.publish(pose)
        aligned = query @ matrix[:3, :3].T + matrix[:3, 3]
        self.scan_pub.publish(cloud_message(aligned, self.cfg['map_frame'], stamp))

    def destroy_node(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Relocalizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
