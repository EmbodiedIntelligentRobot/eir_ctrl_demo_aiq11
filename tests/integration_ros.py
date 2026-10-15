"""Opt-in end-to-end test with an AIQ11 MJCF and ROS DDS; no hardware access."""
import argparse
import os
import math
import json
import re
from pathlib import Path
import signal
import subprocess
import tempfile
import time

os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
os.environ.setdefault('ROS_DOMAIN_ID', '184')

import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile
import yaml
from ehr_message.srv import LocomotionModeSwitch
from ehr_message.srv import MotionCmd
from ehr_message.msg import LocoVelocity
from prime_msgs.msg import DebugRuntime, DebugJointArray, DebugMotorArray
from sensor_msgs.msg import Image
import struct


def pause_gui(pid):
    # Optional GUI test dependency. Send only to the window owned by this child.
    from Xlib import X, XK, display
    from Xlib.ext import xtest
    connection = display.Display()
    try:
        root = connection.screen().root
        atom = connection.intern_atom('_NET_WM_PID')
        pending = list(root.query_tree().children)
        window = None
        while pending:
            candidate = pending.pop()
            value = candidate.get_full_property(atom, X.AnyPropertyType)
            if value is not None and len(value.value) and int(value.value[0]) == pid:
                window = candidate
                break
            pending.extend(candidate.query_tree().children)
        assert window is not None, 'Child MuJoCo window not found'
        code = connection.keysym_to_keycode(XK.string_to_keysym('space'))
        old_focus = connection.get_input_focus()
        pointer = root.query_pointer()
        geometry = window.get_geometry()
        try:
            window.configure(stack_mode=X.Above)
            window.set_input_focus(X.RevertToParent, X.CurrentTime)
            window.warp_pointer(geometry.width // 2, geometry.height // 2)
            connection.sync()
            time.sleep(.1)
            xtest.fake_input(connection, X.KeyPress, code)
            xtest.fake_input(connection, X.KeyRelease, code)
            connection.sync()
            time.sleep(.1)
        finally:
            connection.set_input_focus(old_focus.focus, old_focus.revert_to, X.CurrentTime)
            root.warp_pointer(pointer.root_x, pointer.root_y)
            connection.sync()
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['onnx', 'libtorch'], required=True)
    parser.add_argument('--perception-execution', choices=['combined','split'])
    parser.add_argument('--external-depth-fault', choices=['dropout','invalid'])
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--release-after-ready', action='store_true')
    parser.add_argument('--ready-hold-seconds', type=float, default=0)
    parser.add_argument('--idle-seconds', type=float, default=0)
    parser.add_argument('--idle-only', action='store_true')
    parser.add_argument('--ready-only', action='store_true')
    parser.add_argument('--mode', type=int, choices=[28,30,31,32,33,34,41], default=28)
    parser.add_argument('--complete-motion', action='store_true')
    parser.add_argument('--switch-modes', type=int, nargs='*', default=[])
    parser.add_argument('--switch-seconds', type=float, default=3)
    parser.add_argument('--walk-seconds', type=float, default=2)
    parser.add_argument('--velocity-x', type=float, default=.05)
    parser.add_argument('--velocity-y', type=float, default=0.0)
    parser.add_argument('--velocity-z', type=float, default=0.0)
    parser.add_argument('--velocity-seconds', type=float, default=.5)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--pause-after-idle', action='store_true')
    parser.add_argument('--check-wrist-still', action='store_true')
    parser.add_argument('--expect-limit-warning', action='store_true')
    parser.add_argument('--upper-body-actions', action='store_true')
    parser.add_argument('--burst-requests', type=int, default=0)
    parser.add_argument('--cpu-load', type=int, default=0, help='Local CPU busy workers during RL only')
    parser.add_argument('--metrics', type=Path)
    parser.add_argument('--check-velocity-expiry', action='store_true')
    args = parser.parse_args()
    if args.pause_after_idle and not (args.gui and args.idle_only):
        parser.error('--pause-after-idle requires --gui --idle-only')
    if args.check_wrist_still and not (args.idle_only and args.idle_seconds >= 10):
        parser.error('--check-wrist-still requires --idle-only and --idle-seconds >= 10')
    if args.expect_limit_warning and (args.idle_only or args.idle_seconds <= 0):
        parser.error('--expect-limit-warning requires an idle interval followed by READY/RL')
    source = Path(args.config).resolve()
    config = yaml.safe_load(source.read_text())
    for key in ['robot', 'flight_recorder']:
        config[key] = str(source.parent / config[key])
    config['policies'] = {mode: str(source.parent / profile) for mode, profile in config['policies'].items()}
    if config.get('motion_stream'):
        config['motion_stream'] = str(source.parent / config['motion_stream'])
    config['runtime']['model'] = str(Path(args.model).resolve())
    config['runtime']['backend'] = 'simulation'
    config['runtime']['headless'] = not args.gui
    config['runtime']['suspend'] = True
    config['runtime']['release_after_ready'] = args.release_after_ready
    config['remote']['source'] = 'external'
    config['remote']['gui'] = False
    config['inference']['backend'] = args.backend
    if args.perception_execution:
        config['inference']['perception_execution'] = args.perception_execution
    config['debug']['enabled'] = True
    robot = yaml.safe_load(Path(config['robot']).read_text())
    joint_count, motor_count = len(robot['joint_names']), len(robot['motor_names'])
    wrist_slots = []
    if args.check_wrist_still:
        if not all(name in robot['motor_names'] for name in ['l_arm_51', 'r_arm_51']):
            parser.error('--check-wrist-still requires PRO wrist actuators')
        wrist_slots = [robot['motor_names'].index(name) for name in ['l_arm_51', 'r_arm_51']]
    with tempfile.TemporaryDirectory(prefix='eir-ros-test-') as directory:
        path = Path(directory)
        (path / 'config.yaml').write_text(yaml.safe_dump(config))
        with (path / 'app.log').open('w') as log:
            process = subprocess.Popen([args.executable, str(path / 'config.yaml')], stdout=log, stderr=log)
            rclpy.init()
            node = rclpy.create_node('eir_integration_test')
            client = node.create_client(LocomotionModeSwitch, config['remote']['mode_service'], qos_profile=QoSProfile(depth=64))
            motion_client = None
            if args.upper_body_actions:
                manifest = yaml.safe_load(Path(config['motion_stream']).read_text())
                motion_client = node.create_client(MotionCmd, manifest['service'])
            counts = [0, 0, 0]
            inference_us = []
            motion_control_us, motion_infer_us = [], []
            depth_images = []
            velocity_observed = False
            wrist_samples = 0
            measuring = False
            pressure = []
            measured_min, measured_max = [math.inf]*joint_count, [-math.inf]*joint_count
            target_min, target_max = [math.inf]*joint_count, [-math.inf]*joint_count
            joint_samples = 0
            control_us = []
            wrist_peak_torque = [0., 0.]
            wrist_peak_velocity = [0., 0.]
            # GUI/model loading can take longer than the settling interval. Start
            # measuring only after STILL is confirmed and physics has been running.
            wrist_measure_at = math.inf
            limit_warning_seen = False
            last_runtime = None
            def callback(index):
                def receive(message):
                    nonlocal wrist_samples, joint_samples, last_runtime
                    counts[index] += 1
                    if index == 0:
                        last_runtime = message
                    if index == 0 and measuring:
                        inference_us.append(message.infer_time_us)
                        control_us.append(message.ctrl_loop_duration * 1e6)
                    if index == 0 and args.upper_body_actions and message.ctrl_name == 'AMP_HUMANOID_WALK':
                        motion_control_us.append(message.ctrl_loop_duration * 1e6)
                        motion_infer_us.append(message.infer_time_us)
                    if index == 1:
                        assert len(message.q_act) == message.joint_num == joint_count
                        if measuring:
                            joint_samples += 1
                            for j in range(joint_count):
                                measured_min[j] = min(measured_min[j], message.q_act[j])
                                measured_max[j] = max(measured_max[j], message.q_act[j])
                                target_min[j] = min(target_min[j], message.q_des[j])
                                target_max[j] = max(target_max[j], message.q_des[j])
                    if index == 2:
                        assert len(message.kp) == message.motor_num == motor_count
                        if args.check_wrist_still and time.monotonic() >= wrist_measure_at:
                            wrist_samples += 1
                            for side, slot in enumerate(wrist_slots):
                                torque, velocity = message.tau_act[slot], message.v_act[slot]
                                assert math.isfinite(torque) and math.isfinite(velocity)
                                assert message.kp[slot] == 0
                                assert abs(message.kd[slot] - config['runtime']['damping_kd']) < 1e-6
                                wrist_peak_torque[side] = max(wrist_peak_torque[side], abs(torque))
                                wrist_peak_velocity[side] = max(wrist_peak_velocity[side], abs(velocity))
                                assert abs(torque) < .5 and abs(velocity) < .1, f'Wrist STILL oscillation: side={side} tau={torque} v={velocity}'
                return receive
            subscriptions = [node.create_subscription(kind, config['debug'][topic], callback(i), qos_profile_sensor_data)
                for i, (kind, topic) in enumerate([(DebugRuntime, 'runtime_topic'),
                    (DebugJointArray, 'joint_topic'), (DebugMotorArray, 'motor_topic')])]
            hand_samples, last_hand = 0, None
            robot_config = yaml.safe_load(Path(config['robot']).read_text())
            check_hand = robot_config.get('hand_control', {}).get('enabled', False) and config['debug'].get('hand_topic')
            if check_hand:
                torque_limits = robot_config['simulation_hand']['torque_limits']
                def receive_hand(message):
                    nonlocal hand_samples, last_hand
                    assert message.joint_num == len(message.q_act) == len(message.q_des) == len(message.tau_act) == 12
                    assert all(math.isfinite(v) and 0 <= v <= 1 for v in list(message.q_act) + list(message.q_des))
                    assert message.q_des[5] == message.q_des[11] == 0
                    assert all(math.isfinite(v) and abs(v) <= torque_limits[i//6] + 1e-6
                               for i, v in enumerate(message.tau_act))
                    hand_samples += 1
                    last_hand = message
                subscriptions.append(node.create_subscription(DebugJointArray, config['debug']['hand_topic'],
                    receive_hand, qos_profile_sensor_data))
            if config['debug'].get('depth_image_topic'):
                def depth_image(message):
                    assert (message.height, message.width, message.encoding) == (18, 32, '32FC1')
                    pixels = struct.unpack(('>' if message.is_bigendian else '<') + '576f', bytes(message.data))
                    assert all(math.isfinite(x) and 0 <= x <= 2.5 for x in pixels)
                    depth_images.append((min(pixels), max(pixels)))
                subscriptions.append(node.create_subscription(Image, config['debug']['depth_image_topic'],
                                                               depth_image, qos_profile_sensor_data))
            publisher = node.create_publisher(LocoVelocity, config['remote']['velocity_topic'], qos_profile_sensor_data)
            depth_fault = False
            if args.external_depth_fault:
                from depth_transport_msgs.msg import DepthStack
                depth_publisher = node.create_publisher(DepthStack, config['runtime']['external_depth']['topic'], qos_profile_sensor_data)
                depth_sequence = 0
                def send_depth():
                    nonlocal depth_sequence
                    if depth_fault and args.external_depth_fault == 'dropout': return
                    depth_sequence += 1
                    msg = DepthStack()
                    msg.header.stamp = node.get_clock().now().to_msg()
                    msg.header.frame_id = config['runtime']['external_depth']['frame_id']
                    msg.frame_count, msg.height, msg.width = 8, 18, 32
                    msg.preprocessing_version = 'parkour_depth_v2'
                    msg.source_sequence = depth_sequence
                    msg.data = [float('nan') if depth_fault else .7] * 4608
                    depth_publisher.publish(msg)
                depth_timer = node.create_timer(1/60, send_depth)
            def call(mode):
                request = LocomotionModeSwitch.Request()
                request.cmd = mode
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=3)
                assert future.done(), 'mode service did not respond'
                return future.result()
            def healthy(mode):
                nonlocal limit_warning_seen
                assert process.poll() is None, 'Application exited unexpectedly'
                response = call(0)
                assert response.mode == mode and 'FAULT' not in response.fb_msg, response.fb_msg
                limit_warning_seen |= 'warning:' in response.fb_msg
                return response
            def motion_call(cmd, expected=0):
                request = MotionCmd.Request(cmd=cmd, session_id='offline-test')
                future = motion_client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=3)
                assert future.done(), 'motion service did not respond'
                result = future.result()
                assert result.err_code == expected and result.result == (expected == 0), result
            def motion_status():
                message = healthy(28).fb_msg
                match = re.search(r'upper_motion_cmd=(\d+) upper_motion_phase=(\d+) upper_motion_seconds=([0-9.]+)', message)
                assert match, message
                return int(match[1]), int(match[2]), float(match[3])
            def wait_motion_idle(timeout):
                deadline = time.monotonic() + timeout
                phases = set()
                while True:
                    cmd, phase, elapsed = motion_status()
                    phases.add(phase)
                    if phase == 0:
                        assert cmd == 0
                        break
                    assert time.monotonic() < deadline, f'Motion did not finish: {cmd} {phase} {elapsed}'
                    rclpy.spin_once(node, timeout_sec=.02)
                return phases
            def hand_goal(expected, timeout=3):
                if not check_hand: return
                deadline = time.monotonic() + timeout
                while last_hand is None or any(abs(a-b) > 1e-5 for a,b in zip(last_hand.q_des, expected)):
                    assert time.monotonic() < deadline, f'Hand target did not settle: {last_hand}'
                    healthy(28)
                    rclpy.spin_once(node, timeout_sec=.02)
            try:
                assert client.wait_for_service(timeout_sec=15), 'mode service unavailable'
                assert 'suspended=true' in healthy(1).fb_msg
                if args.burst_requests:
                    for modes in ([0] * args.burst_requests,
                                  [2 if i % 2 else 1 for i in range(args.burst_requests)]):
                        futures = []
                        for mode in modes:
                            request = LocomotionModeSwitch.Request(); request.cmd = mode
                            futures.append(client.call_async(request))
                        deadline = time.monotonic() + 5
                        while not all(future.done() for future in futures):
                            assert time.monotonic() < deadline, f'Burst replies missing: {sum(f.done() for f in futures)}/{len(futures)}'
                            rclpy.spin_once(node, timeout_sec=.01)
                        assert all(f.result() is not None for f in futures)
                        print(f'Burst modes={set(modes)}: {len(futures)}/{len(futures)} replies', flush=True)
                    assert call(1).result
                    healthy(1)
                if motion_client:
                    assert motion_client.wait_for_service(timeout_sec=3)
                    motion_call(11, 3)  # Never take over STILL/READY.
                    motion_call(3, 1)   # No pause/realtime stream in this minimal port.
                    motion_call(255, 2)
                wrist_measure_at = time.monotonic() + 5
                deadline = time.monotonic() + args.idle_seconds
                while time.monotonic() < deadline:
                    healthy(1)
                    rclpy.spin_once(node, timeout_sec=.05)
                    time.sleep(.05)
                if args.expect_limit_warning:
                    assert limit_warning_seen, 'Idle posture did not exercise the joint limit warning'
                if args.idle_only:
                    assert all(counts), f'Missing debug topics: {counts}'
                    if args.check_wrist_still:
                        assert wrist_samples >= 20, f'Insufficient wrist samples: {wrist_samples}'
                        print(f'Wrist STILL: samples={wrist_samples}, peak_torque_Nm={wrist_peak_torque}, peak_velocity_rad_s={wrist_peak_velocity}')
                    print(f'{args.backend}: GUI={args.gui}, STILL healthy for {args.idle_seconds}s; debug counts={counts}')
                    if args.pause_after_idle:
                        pause_gui(process.pid)
                        deadline = time.monotonic() + 3
                        while True:
                            response = call(0)
                            if 'FAULT (latched)' in response.fb_msg:
                                break
                            assert time.monotonic() < deadline, 'Pause did not trigger feedback watchdog'
                            time.sleep(.02)
                        assert 'Feedback timeout' in response.fb_msg, response.fb_msg
                        for mode in [1, 2, 28]:
                            response = call(mode)
                            assert not response.result and 'FAULT' in response.fb_msg, response.fb_msg
                        print(f'GUI pause: feedback fault reported and remains latched: {call(0).fb_msg}')
                        # Keep the ROS executor alive: paused physics must not turn
                        # old snapshots into new telemetry with new header stamps.
                        drain_until = time.monotonic() + .35
                        while time.monotonic() < drain_until:
                            rclpy.spin_once(node, timeout_sec=.02)
                        before, before_hands = counts.copy(), hand_samples
                        frozen_sequence = last_runtime.feedback_sequence
                        frozen_stamp = last_runtime.feedback_stamp_ns
                        until = time.monotonic() + 1
                        while time.monotonic() < until:
                            rclpy.spin_once(node, timeout_sec=.02)
                        assert counts[1:] == before[1:] and hand_samples == before_hands
                        assert counts[0] > before[0]
                        assert last_runtime.ctrl_name == 'FAULT' and last_runtime.controller_fault != 0
                        assert not last_runtime.feedback_valid and last_runtime.feedback_age_ns > 100_000_000
                        assert last_runtime.feedback_sequence == frozen_sequence
                        assert last_runtime.feedback_stamp_ns == frozen_stamp
                        print(f'GUI stale telemetry: body/hand stopped, runtime +{counts[0]-before[0]}, '
                              f'sequence={frozen_sequence}, age_ns={last_runtime.feedback_age_ns}, '
                              f'controller_fault={last_runtime.controller_fault}')
                    return
                assert not call(args.mode).result, 'RL must require READY'
                assert not call(22).result, 'Unsupported mode must be rejected'
                assert call(2).result
                assert 'suspended=true' in healthy(2).fb_msg, 'Released before READY transition finished'
                deadline = time.monotonic() + 12
                while True:
                    response = healthy(2)
                    if 'ready=true' in response.fb_msg and (not args.release_after_ready or 'suspended=false' in response.fb_msg):
                        break
                    assert time.monotonic() < deadline, 'READY arrival timeout'
                    rclpy.spin_once(node, timeout_sec=.1)
                    time.sleep(.05)
                if args.expect_limit_warning:
                    assert 'warning:' not in response.fb_msg, 'Limit warning did not clear in READY'
                    print('Limit warning observed during STILL; READY recovered without FAULT')
                assert ('suspended=false' if args.release_after_ready else 'suspended=true') in response.fb_msg
                print('READY completion: ' + response.fb_msg)
                deadline = time.monotonic() + args.ready_hold_seconds
                while time.monotonic() < deadline:
                    response = healthy(2)
                    assert ('suspended=false' if args.release_after_ready else 'suspended=true') in response.fb_msg
                    rclpy.spin_once(node, timeout_sec=.05)
                assert 'ready=true' in response.fb_msg, 'READY posture lost during hold'
                if args.ready_only:
                    assert call(1).result
                    healthy(1)
                    assert all(counts), f'Missing debug topics: {counts}'
                    print(f'{args.backend}: STILL -> READY ({args.ready_hold_seconds}s hold) -> STILL; debug counts={counts}')
                    return
                # READY is a live posture check. Release can briefly invalidate
                # it between query and request; retry within a fixed deadline,
                # preserving the controller's gate and rejecting any real fault.
                transition_deadline = time.monotonic() + 2
                rejected = 0
                transition = call(args.mode)
                while not transition.result:
                    current = healthy(2)
                    assert time.monotonic() < transition_deadline, 'READY did not permit RL within 2s: ' + current.fb_msg
                    rejected += 1
                    rclpy.spin_once(node,timeout_sec=.02)
                    transition = call(args.mode)
                if rejected:
                    print(f'READY -> RL safely retried after {rejected} live posture rejections',flush=True)
                if args.check_velocity_expiry:
                    observations = []
                    for trial in range(10):
                        lease = .1
                        until = time.monotonic() + .2
                        last_publish = 0
                        while time.monotonic() < until:
                            publisher.publish(LocoVelocity(linear_vx=.15, keep_time=lease))
                            last_publish = time.monotonic()
                            rclpy.spin_once(node, timeout_sec=.005)
                        assert 'input_vx=0.150000' in healthy(args.mode).fb_msg
                        deadline = last_publish + config['timing']['velocity_timeout'] + .5
                        while True:
                            current = healthy(args.mode)
                            if 'input_vx=0.000000' in current.fb_msg: break
                            assert time.monotonic() < deadline, 'Expired velocity still enters model'
                            rclpy.spin_once(node, timeout_sec=.001)
                        observations.append((time.monotonic() - last_publish - lease) * 1000)
                    print(f'Velocity expiry observation ms after publish+lease: n=10 max={max(observations):.3f} '
                          f'p50={sorted(observations)[5]:.3f}; includes DDS and query delay', flush=True)
                pressure = [subprocess.Popen(['/usr/bin/python3', '-c',
                    'while True: sum(i*i for i in range(10000))']) for _ in range(args.cpu_load)]
                measuring = True
                started = time.monotonic()
                deadline = started + args.walk_seconds
                completed_at = None
                while time.monotonic() < deadline:
                    # A short velocity lease must expire; do not continuously renew it.
                    if args.mode in (28, 41) and time.monotonic() < started + args.velocity_seconds:
                        msg = LocoVelocity()
                        msg.priority = LocoVelocity.PRIORITY_REMOTE
                        msg.linear_vx = args.velocity_x
                        msg.linear_vy = args.velocity_y
                        msg.angular_wz = args.velocity_z
                        msg.keep_time = .1
                        publisher.publish(msg)
                    rclpy.spin_once(node, timeout_sec=.05)
                    current = healthy(args.mode)
                    if args.mode == 41:
                        match = re.search(r'depth_age_ms=([0-9.]+)', current.fb_msg)
                        assert match and float(match.group(1)) < 100, current.fb_msg
                        if args.perception_execution == 'split':
                            assert 'execution=split' in current.fb_msg, current.fb_msg
                            age = re.search(r'feature_age_at_action_ms=([0-9.]+)',current.fb_msg)
                            assert age and float(age.group(1)) <= 100, current.fb_msg
                        velocity = re.search(r'input_vx=([-0-9.]+)', current.fb_msg)
                        if velocity and abs(float(velocity.group(1)) - args.velocity_x) < 1e-5:
                            velocity_observed = True
                    if args.complete_motion:
                        match = re.search(r'motion_frame=(\d+)/(\d+)', current.fb_msg)
                        assert match is not None, current.fb_msg
                        frame, count = map(int, match.groups())
                        if count and frame == count - 1:
                            if completed_at is None:
                                completed_at = time.monotonic()
                                print('Final reference frame: ' + current.fb_msg, flush=True)
                            if time.monotonic() - completed_at >= 2:
                                break
                if args.complete_motion:
                    assert completed_at is not None, 'Motion did not reach its final reference frame'
                    assert time.monotonic() - completed_at >= 2, 'Insufficient final-frame hold'
                if args.mode == 41 and args.velocity_seconds > 0:
                    assert velocity_observed, 'Velocity command never reached perception inference'
                    print(f'Perception observed input_vx={args.velocity_x}', flush=True)
                if config['debug'].get('depth_image_topic'):
                    assert len(depth_images) >= 5, 'Depth image topic missing'
                    print(f'Depth images: n={len(depth_images)} latest metres={depth_images[-1]}', flush=True)
                if inference_us:
                    samples = sorted(inference_us)
                    print(f'Policy inference us: n={len(samples)} p50={samples[len(samples)//2]} p99={samples[int(.99*(len(samples)-1))]} max={max(samples)}', flush=True)
                measuring = False
                if args.metrics:
                    stats = lambda values: {'n': len(values), 'p50': sorted(values)[len(values)//2],
                        'p99': sorted(values)[int(.99*(len(values)-1))], 'max': max(values)} if values else {}
                    data = {'backend': args.backend, 'cpu_workers': args.cpu_load,
                            'duration_seconds': time.monotonic() - started,
                            'joint_samples': joint_samples, 'inference_us': stats(inference_us),
                            'control_us': stats(control_us), 'joints': []}
                    for j, name in enumerate(robot['joint_names']):
                        data['joints'].append({'name': name, 'measured_min': measured_min[j],
                            'measured_max': measured_max[j], 'target_min': target_min[j],
                            'target_max': target_max[j], 'lower': robot['lower'][j], 'upper': robot['upper'][j]})
                    args.metrics.write_text(json.dumps(data, indent=2, allow_nan=False))
                print(f'Active duration: {time.monotonic() - started:.2f}s', flush=True)
                print('Final policy diagnostics: ' + healthy(args.mode).fb_msg, flush=True)
                if motion_client:
                    hand_goal(robot_config['hand_control']['closed_position'])
                    for cmd in [11, 12, 13, 14]:
                        motion_call(cmd)
                        assert motion_status()[:2] == (cmd, 1)
                        hand_goal([0]*12) # Current four source motions explicitly open all active fingers.
                        phases = wait_motion_idle(17)
                        assert {1, 2, 3, 0} <= phases, phases
                        hand_goal(robot_config['hand_control']['closed_position'])
                        print(f'Upper-body action {cmd}: completed with recovery, mode remains AMP', flush=True)
                    motion_call(11)
                    deadline = time.monotonic() + 1
                    while time.monotonic() < deadline:
                        msg = LocoVelocity(linear_vx=.05, keep_time=.1)
                        publisher.publish(msg)
                        healthy(28)
                        rclpy.spin_once(node, timeout_sec=.02)
                    motion_call(14)  # Restart from the previous emitted command.
                    assert motion_status()[:2] == (14, 1)
                    motion_call(2)
                    assert {2, 3, 0} <= wait_motion_idle(3)
                    motion_call(2)  # Idempotent STOP.
                    motion_call(11)
                    assert call(30).result
                    response = healthy(30)
                    assert 'upper_motion_cmd=0 upper_motion_phase=0' in response.fb_msg
                    motion_call(11, 3)
                    assert call(28).result
                    deadline = time.monotonic() + 1
                    while time.monotonic() < deadline:
                        healthy(28)
                        rclpy.spin_once(node, timeout_sec=.02)
                    print('Upper-body restart/STOP/mode cancellation passed', flush=True)
                    if check_hand:
                        assert hand_samples > 100
                        print(f'Three-finger targets open during actions and recover closed; '
                              f'finite bounded feedback/torque: {hand_samples} debug samples', flush=True)
                    for label, values in [('control', motion_control_us), ('inference', motion_infer_us)]:
                        values.sort()
                        assert values, f'Missing {label} timings'
                        print(f'Upper-body {label} us: n={len(values)} p50={values[len(values)//2]:.1f} '
                              f'p99={values[int(.99*(len(values)-1))]:.1f} max={max(values):.1f}', flush=True)
                assert all(counts), f'Missing debug topics: {counts}'
                for mode in args.switch_modes:
                    assert call(mode).result, f'Direct switch to {mode} rejected'
                    if motion_client and mode != 28:
                        motion_call(11, 3)
                    deadline = time.monotonic() + args.switch_seconds
                    while time.monotonic() < deadline:
                        healthy(mode)
                        rclpy.spin_once(node, timeout_sec=.05)
                    print(f'Direct switch to {mode}: healthy for {args.switch_seconds}s', flush=True)
                # A forbidden READY request must neither stop nor reset the policy.
                active_mode = args.switch_modes[-1] if args.switch_modes else args.mode
                response = call(2)
                assert not response.result and response.mode == active_mode, response.fb_msg
                assert 'READY is only allowed from STILL' in response.fb_msg, response.fb_msg
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    healthy(active_mode)
                    rclpy.spin_once(node, timeout_sec=.05)
                print('RL -> READY rejected; current policy remains healthy', flush=True)
                if args.external_depth_fault:
                    depth_fault = True
                    deadline = time.monotonic() + 1
                    while True:
                        response = call(0)
                        if 'FAULT (latched)' in response.fb_msg: break
                        assert time.monotonic() < deadline, 'Depth failure did not stop inference'
                        rclpy.spin_once(node, timeout_sec=.01)
                    assert 'External depth' in response.fb_msg, response.fb_msg
                    assert 'fault=2' in response.fb_msg, response.fb_msg
                    for mode in [1,2,28,41]: assert not call(mode).result
                    print('Depth safety fault latched: ' + response.fb_msg, flush=True)
                    return
                if motion_client:
                    motion_call(11)
                    assert motion_status()[:2] == (11, 1)
                response = call(1)
                assert response.result and response.mode == 1
                assert ('suspended=false' if args.release_after_ready else 'suspended=true') in healthy(1).fb_msg
                if motion_client:
                    assert 'upper_motion_cmd=0 upper_motion_phase=0' in healthy(1).fb_msg
                    motion_call(11, 3)
                print(f'{args.backend}: GUI={args.gui}, STILL -> READY -> RL mode={args.mode} ({args.walk_seconds}s) -> STILL; debug counts={counts}')
            except BaseException:
                log.flush()
                print((path / 'app.log').read_text()[-6000:])
                raise
            finally:
                for worker in pressure: worker.terminate()
                for worker in pressure: worker.wait(timeout=5)
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                node.destroy_node()
                rclpy.shutdown()
                if args.log:
                    log.flush()
                    args.log.write_text((path / 'app.log').read_text())
                assert process.returncode == 0, f'Application exit code: {process.returncode}'


if __name__ == '__main__':
    main()
