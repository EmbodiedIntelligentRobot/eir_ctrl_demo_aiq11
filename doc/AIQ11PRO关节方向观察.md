# AIQ11 PRO 关节方向观察

在可靠支撑或悬吊下手动改变关节姿态，观察 32 个身体关节反馈。`--observe-joints` 从第一条应用命令起强制所有身体电机 Kp、Kd、目标速度和前馈力矩为零，禁止 READY、行走和 MIMIC，关闭手部命令、离线动作及 SBUS 快捷键。ROS 查询和 STILL 可用，查询包含 `joint_observation=true`。反馈超时、非法数据及故障锁定继续生效；退出和故障仍执行 SDK 急停，其内部增益不由本选项覆盖。

## 机器人端

若已有进程，先在其终端按 Ctrl+C，再运行：

```bash
cd /home/core-aiq11/Data/eir_aiq11_pro_hardware
./run.sh --observe-joints
```

脚本自动使用 sudo。该模式仍初始化 EtherCAT 和发送零增益命令，不提供机体支撑。退出按 Ctrl+C；后续不带参数的 `./run.sh` 使用原有正常阻尼配置。

源码构建入口为 `eir_ctrl_demo_aiq11 --observe-joints config/pro_hardware.yaml`。无硬件初始化检查为 `eir_ctrl_demo_aiq11 --observe-joints --check-config config/pro_hardware.yaml`，应列出全部 32 槽 Kd=0。

## 本地 RViz

本次开发机已准备好脚本，桌面终端执行：

```bash
/tmp/eir-aiq11-rviz/view.sh
```

脚本启动本地 `robot_state_publisher`、基座姿态 TF 转换和 RViz，关闭 RViz 时清理本脚本启动的两个后台节点。使用 ROS Humble、`ROS_DOMAIN_ID=0`、`ROS_LOCALHOST_ONLY=0`、Fast DDS。应用发布 `/joint_states`（`sensor_msgs/msg/JointState`）和 `/base_orientation`（`geometry_msgs/msg/QuaternionStamped`），均为 best effort、当前 20 Hz。本地模型和 TF 使用专用话题 `/aiq11_joint_view/robot_description`、`/aiq11_joint_view/tf`、`/aiq11_joint_view/tf_static`，避免局域网其他机器人覆盖模型或同名坐标系。

RViz 固定坐标系为 `imu_world`。本地脚本把 `/base_orientation` 转换为 `imu_world → base_link`，机身及所有下游关节随 IMU 姿态旋转。四元数沿用控制反馈的归一化与 `base_from_imu` 安装转换，不重复转换方向；时间戳与关节数据对应同一反馈接收时刻。`imu_world` 是 IMU 姿态参考系，平移固定为零，不表示定位结果、地面高度或绝对航向标定，也不对加速度积分。该话题只提供显示所需姿态，不是完整 IMU 原始数据。

也可在本地三个终端分别执行（都先加载环境）：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 终端一
ros2 run robot_state_publisher robot_state_publisher --ros-args \
  -r robot_description:=/aiq11_joint_view/robot_description \
  -r /tf:=/aiq11_joint_view/tf -r /tf_static:=/aiq11_joint_view/tf_static \
  --params-file /tmp/eir-aiq11-rviz/robot_state_publisher.yaml

# 终端二，先执行上面的 source/export
/usr/bin/python3 /tmp/eir-aiq11-rviz/base_orientation_tf.py --ros-args \
  -r /tf:=/aiq11_joint_view/tf

# 终端三，先执行上面的 source/export
rviz2 -d /tmp/eir-aiq11-rviz/joints.rviz --ros-args \
  -r /tf:=/aiq11_joint_view/tf -r /tf_static:=/aiq11_joint_view/tf_static
```

不启动 `joint_state_publisher` 或其 GUI，避免生成的角度覆盖实测反馈。可用 `ros2 topic echo /joint_states --once --qos-reliability best_effort` 和 `ros2 topic echo /base_orientation --once --qos-reliability best_effort` 查看输入；本地和机器人需在能发现 ROS 2 节点的局域网。旧程序不发布基座姿态，更新后须重启机器人程序和本地 `view.sh`。未收到有效姿态时不伪造单位四元数；仅查看关节相对运动时，可手动把 RViz 的 Fixed Frame 改回 `base_link`。

默认选中 `Move Camera`：左键拖动旋转视角，中键拖动平移，滚轮缩放。显示帧率为 20 FPS，与当前反馈发布频率一致。配置必须保留相机工具；Humble RViz 在工具列表为空时无法拖拽，并可能在键盘输入时段错误。`Stereo is NOT SUPPORTED` 是普通渲染信息。

应用开启 `debug.enabled` 时，正常模式也发布这两个话题。发布在普通 ROS 线程执行；过期反馈停止发布，TF 转换也不刷新旧时间戳，RViz 最后保留的姿态不代表新反馈。

腰部 `torso_3`（俯仰）与 `torso_2`（横滚）使用两个腰部电机的 SDK 并联正解，与旧 PRO 程序一致；独立编码器读数不直接作为显示角度。SDK 输入顺序为 `[torso_33, torso_31]`，输出顺序为 `[俯仰, 横滚]`。脚踝、颈部同样使用相应 SDK 正解，腰部 yaw `torso_1` 直接使用电机反馈。

## 在其他工作站准备模型

安装 ROS Humble 的 rviz2、robot_state_publisher、rclpy、tf2_ros 和 Python YAML。从机器人取回原机的 `aiq11_pro_pin.urdf`、`aiq11_pro.urdf`；本地 SDK 提供对应网格。应用目录执行：

```bash
python3 tools/prepare_joint_rviz.py \
  --urdf /本地路径/aiq11_pro_pin.urdf \
  --visual-urdf /本地路径/aiq11_pro.urdf \
  --meshes ../../sdk/share/eir_mujoco/aiq11_pro/model/meshes \
  --output /tmp/eir-aiq11-rviz
```

原机控制 URDF 固定了颈部三关节；工具确认父子坐标系一致后，使用原可视化 URDF 的定义恢复这三关节。其余 29 个活动身体关节保持原定义，网格解析到本地绝对路径。活动关节集合必须与应用 32 项名称完全一致。

本轮使用注入后端、软件与仿真回归，并在隔离 ROS 域检查 32 个活动 TF、颈部方向和无显示器 RViz 模型加载。没有代用户启动真机，跨机实时显示和实物方向由用户操作确认。

整机姿态显示的本地验证：ROS 诊断覆盖非单位 IMU 四元数、非平凡安装旋转、正确 wxyz 转换及反馈断流停止发布。`tests/rviz_orientation_test.py` 验证专用 TF、无效四元数拒绝、时间戳保持及停止输入后不重复发送；完整模型检查共 33 个动态 TF（基座加 32 关节），施加 0.4 rad 俯仰时腿部完整链随基座旋转，RViz 正常加载。以上均为本机软件验证，不代表真机方向已确认。
