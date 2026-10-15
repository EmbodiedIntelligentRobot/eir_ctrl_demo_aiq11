#!/usr/bin/env python3
"""ROS integration check for the local orientation viewer; no hardware access."""
import math
from pathlib import Path
import sys
import time
import unittest

import rclpy
from geometry_msgs.msg import QuaternionStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from base_orientation_tf import BaseOrientationTf


class OrientationView(unittest.TestCase):
    def test_measured_rotation_only_on_private_tf(self):
        rclpy.init(args=["--ros-args", "-r", "/tf:=/aiq11_joint_view/tf"])
        bridge = BaseOrientationTf()
        probe = rclpy.create_node("orientation_view_test")
        executor = SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.add_node(probe)
        messages = []
        subscription = probe.create_subscription(
            TFMessage, "/aiq11_joint_view/tf", messages.append, qos_profile_sensor_data)
        publisher = probe.create_publisher(QuaternionStamped, "base_orientation", qos_profile_sensor_data)

        def spin(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.01)

        def send(x, y, z, w, frame="imu_world"):
            message = QuaternionStamped()
            message.header.frame_id = frame
            message.header.stamp = probe.get_clock().now().to_msg()
            message.quaternion.x, message.quaternion.y = x, y
            message.quaternion.z, message.quaternion.w = z, w
            publisher.publish(message)
            spin(.15)
            return message

        try:
            deadline = time.monotonic() + 5
            while (not publisher.get_subscription_count() or
                   not bridge.broadcaster.pub_tf.get_subscription_count()) and time.monotonic() < deadline:
                spin(.05)
            self.assertGreater(publisher.get_subscription_count(), 0)
            self.assertGreater(bridge.broadcaster.pub_tf.get_subscription_count(), 0)
            send(0., 0., 0., 0.)
            send(math.nan, 0., 0., 1.)
            send(0., 0., 0., 1., "base_link")
            self.assertFalse(messages)
            # A 90-degree pitch, scaled by 2, must rotate the whole base about Y.
            sent = send(0., math.sqrt(2), 0., math.sqrt(2))
            self.assertTrue(messages)
            transform = messages[-1].transforms[0]
            self.assertEqual(transform.header, sent.header)
            self.assertEqual(transform.child_frame_id, "base_link")
            rotation = transform.transform.rotation
            self.assertAlmostEqual(rotation.y, math.sqrt(.5))
            self.assertAlmostEqual(rotation.w, math.sqrt(.5))
            self.assertEqual((rotation.x, rotation.z), (0., 0.))
            translation = transform.transform.translation
            self.assertEqual((translation.x, translation.y, translation.z), (0., 0., 0.))
            count = len(messages)
            spin(.3)
            self.assertEqual(len(messages), count)  # No timer invents fresh IMU data.
        finally:
            executor.shutdown()
            probe.destroy_subscription(subscription)
            probe.destroy_node()
            bridge.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
