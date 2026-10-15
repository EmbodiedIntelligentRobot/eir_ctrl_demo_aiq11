#!/usr/bin/env python3
"""Local RViz rotation from measured base orientation; no position estimation."""
import math

import rclpy
from geometry_msgs.msg import QuaternionStamped, TransformStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


class BaseOrientationTf(Node):
    def __init__(self):
        super().__init__("aiq11_base_orientation_viewer")
        self.broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            QuaternionStamped, "base_orientation", self.orientation, qos_profile_sensor_data)

    def orientation(self, message):
        q = message.quaternion
        values = (q.x, q.y, q.z, q.w)
        norm = math.hypot(*values)
        if message.header.frame_id != "imu_world" or not math.isfinite(norm) or norm < 1e-6:
            return
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = "base_link"
        q = transform.transform.rotation
        q.x, q.y, q.z, q.w = (v / norm for v in values)
        # Translation stays zero. Publish only on measured input; never refresh stale poses.
        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init()
    node = BaseOrientationTf()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
