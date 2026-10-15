"""Synthetic raw -> actual processor -> actual stack over local DDS. No device access."""
import argparse
from array import array
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from depth_transport_msgs.msg import DepthStack
from eir_loco_depth_camera.depth_image_processor import DepthImageProcessor
from eir_loco_depth_camera.depth_stack_publisher import DepthStackPublisher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    rclpy.init(args=['--ros-args','--params-file',args.config])
    processor, stack = DepthImageProcessor(), DepthStackPublisher()
    probe = rclpy.create_node('camera_contract_probe')
    executor = SingleThreadedExecutor()
    for node in (processor, stack, probe): executor.add_node(node)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    pub = probe.create_publisher(Image, '/gemini336_a/depth/image_raw', qos)
    info_pub = probe.create_publisher(CameraInfo, '/gemini336_a/depth/camera_info', qos)
    calibration = CameraInfo()
    calibration.header.frame_id = 'gemini336_a_depth_optical_frame'
    calibration.width, calibration.height = 640, 480
    calibration.k = [320., 0., 319.5, 0., 320., 239.5, 0., 0., 1.]
    calibration.distortion_model = 'plumb_bob'
    calibration.d = [0.] * 5
    stamps, stacks, images, fulls = set(), [], [], []
    fault = ''
    raw_buffers = {bad: array('B', np.full((480,640),0 if bad else 1000,dtype='<u2').tobytes()) for bad in [False,True]}
    def raw():
        m = Image()
        stamp = probe.get_clock().now().nanoseconds
        if fault == 'stale': stamp -= 1_000_000_000
        if fault == 'future': stamp += 1_000_000_000
        m.header.stamp.sec, m.header.stamp.nanosec = divmod(stamp,1_000_000_000)
        stamps.add(stamp)
        m.header.frame_id = 'gemini336_a_depth_optical_frame'
        m.width, m.height, m.step, m.encoding = 640, 480, 1280, '16UC1'
        m.data = raw_buffers[fault == 'invalid']
        info_pub.publish(calibration)
        pub.publish(m)
    def image(m, out, shape):
        assert (m.height,m.width,m.encoding)==(*shape,'32FC1')
        assert m.header.stamp.sec*1_000_000_000+m.header.stamp.nanosec in stamps
        np.testing.assert_allclose(np.frombuffer(m.data,dtype='<f4'),.4,atol=1e-6)
        out.append(m)
    def depth(m):
        assert (m.frame_count,m.height,m.width,len(m.data))==(8,18,32,4608)
        assert m.preprocessing_version=='parkour_depth_v2'
        assert m.header.frame_id=='gemini336_a_depth_optical_frame'
        stamp = m.header.stamp.sec*1_000_000_000+m.header.stamp.nanosec
        assert stamp in stamps
        assert 0 <= probe.get_clock().now().nanoseconds-stamp < 100_000_000
        np.testing.assert_allclose(m.data,.4,atol=1e-6) # no second normalization
        stacks.append(m)
    subscriptions = [
        probe.create_subscription(Image,'/gemini336_a/depth/image_processed',lambda m:image(m,images,(18,32)),qos),
        probe.create_subscription(Image,'/gemini336_a/depth/image_processed_full',lambda m:image(m,fulls,(40,64)),qos),
        probe.create_subscription(DepthStack,'/perception/depth_stack',depth,qos)]
    timer = probe.create_timer(1/60,raw)
    def spin(seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline: executor.spin_once(timeout_sec=.01)
    try:
        spin(3)
        assert len(stacks)>20 and len(images)>40 and len(fulls)>40, (len(stacks),len(images),len(fulls))
        for name in ['invalid','stale','future']:
            fault=name
            # Discard in-flight valid messages then require policy stream silence.
            spin(.2); before=len(stacks); spin(.2)
            assert len(stacks)==before, name
        timer.cancel(); spin(.2); before=len(stacks); spin(.2)
        assert len(stacks)==before
        print(f'PASS: raw acquisition stamps preserved; images={len(images)} full={len(fulls)} stacks={len(stacks)}; invalid/stale/future/dropout withheld')
    finally:
        executor.shutdown()
        for node in (probe,stack,processor): node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
