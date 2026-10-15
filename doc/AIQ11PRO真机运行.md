# AIQ11 PRO 真机运行

本次按用户确认的“AIQ11 Pro 底层 SDK 和真机已验证”接入当前应用。新增的是应用到公共 runtime/SDK 的配置与反馈转换；本轮执行假后端软件测试、MuJoCo 回归及 SSH 只读配置核对，没有在机器人上启动新控制器或发送运动命令。底层硬件验证结论由部署方提供，不记作本轮实测。

`config/pro_hardware.yaml` 启用 STILL、READY、AMP WALK、MIMIC A–E 和 PRO LEGACY 三指控制，推理默认 ONNX。无参数入口仍是 9500 仿真。硬件配置不加载 MJCF、虚拟悬吊或仿真手指标定，不启用感知、离线动作和特殊 MIMIC。

## 编译

从工作空间根目录执行。使用与运控主机架构、公共头文件 ABI 一致的完整 SDK；硬件依赖为 EtherLab、yaml-cpp、libssh，其他依赖见应用 README。本轮构建平台为 Linux x86_64 / ROS 2 Humble。

```bash
source /opt/ros/humble/setup.bash
colcon --log-base build/aiq11_hw_interfaces_log build \
  --base-paths apps/eir_ctrl_demo_aiq11/interfaces \
  --build-base build/aiq11_hw_interfaces \
  --install-base install/aiq11_hw_interfaces \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/aiq11_hw_interfaces/setup.bash

cmake -S . -B build/aiq11_hw \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIR_BUILD_CTRL=ON \
  -DEIR_WITH_HARDWARE=ON -DEIR_HARDWARE_FAMILY=aiq11 \
  -DEIR_WITH_MUJOCO=OFF \
  -DEIR_CTRL_WITH_TORCH=OFF -DEIR_CTRL_WITH_ONNX=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build/aiq11_hw --parallel 3 --target eir_ctrl_demo_aiq11
```

使用已有消息包时，可 source 其安装目录代替上面的 colcon 命令，但需包含本版本的诊断字段。其他位置的完整 SDK 使用 `-DEIR_SDK_ROOT=/绝对路径/sdk`。切换为 LibTorch 时需启用对应编译选项并将应用 YAML 的 `inference.backend` 改为 `libtorch`；SDK 硬件选择与推理后端是两个独立配置。

## 配置和启动

`AIQ11_PRO_HARDWARE_CONFIG` 指向这台机器已验证的底层 SDK YAML，使用绝对路径。仓库自带 `sdk/share/eir_base_sdk/config/humanoid.yaml` 默认选择 SPORT，不能直接作为 PRO 配置使用。应用不会自动改机型、零偏、方向、传动比或电机标定。

SDK 配置须选择 `RobotVersion: aiq11-pro`，提供 32 个启用的电机及两个腰部编码器。三指控制要求 `Versions.aiq11-pro.EndEffectorMapping.type: LEGACY`，与 runtime 原有检查一致。当前硬件应用只接受 PRO，其他机型不沿用该配置。

这台机器没有手指，`config/pro_hardware.yaml` 显式设置 `runtime.hand_control: false`。此配置关闭手部动作、手部命令发送和手部故障门控，手部数据为空、离线或无效都不会阻止身体进入 READY/策略模式；身体电机、IMU、腰部编码器、反馈/命令超时和故障锁定继续生效。启动及 `--check-config` 会显示 `Three-finger hand control: disabled`。腰部关节限位警告与手部功能独立，不因此放宽限位。

安装三指手的机器可设置 `runtime.hand_control: true`，沿用机器人配置中的手部动作参数、SDK 类型校验及双手有效反馈保护。省略此字段时保留原机器人配置行为；不会根据手部是否在线自动切换。该覆盖项仅用于硬件配置，仿真继续使用原手部设置。正常启动仍为 `./run.sh`，配置需重启程序后生效。

2026-10-10 通过 SSH 核对：这台机器的启动脚本实际传入 `/eir/locomotion/config/bridge/aiq11_pro/sdk.yaml`，其中机型、32 槽和腰部编码器均符合应用映射；旧桥接程序采用三指百分比接口，但 SDK YAML 尚未显式声明 `LEGACY`。以下命令在机器人上生成新应用专用副本，只补齐三指类型，不覆盖旧配置或改变标定。需要 Python 3 的 `yaml` 模块（Ubuntu 包 `python3-yaml`）。

```bash
export AIQ11_PRO_HARDWARE_CONFIG="$HOME/.config/eir_ctrl_demo_aiq11/pro_sdk.yaml"
python3 - <<'PY'
import os
from pathlib import Path
import yaml

source = Path('/eir/locomotion/config/bridge/aiq11_pro/sdk.yaml')
destination = Path(os.environ['AIQ11_PRO_HARDWARE_CONFIG'])
document = yaml.safe_load(source.read_text())
sdk = document['humanoid_hardware_node']
if sdk['RobotVersion'] != 'aiq11-pro':
    raise SystemExit('配置未选择 aiq11-pro，请核对源文件')
profile = sdk['Versions']['aiq11-pro']
ee = profile.setdefault('EndEffectorMapping', {})
if ee.get('type', 'LEGACY') != 'LEGACY':
    raise SystemExit('已有其他末端类型，请核对源文件')
ee['type'] = 'LEGACY'
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    if yaml.safe_load(destination.read_text()) != document:
        raise SystemExit('目标已存在且内容不同，请核对；未覆盖')
else:
    with destination.open('x') as output:
        yaml.safe_dump(document, output, sort_keys=False)
print(destination)
PY

# 只读配置和解算资源，不构造硬件后端、不调用 Init、不打开 EtherCAT。
build/aiq11_hw/apps/eir_ctrl_demo_aiq11/eir_ctrl_demo_aiq11 \
  --check-config apps/eir_ctrl_demo_aiq11/config/pro_hardware.yaml
```

核对时机器人 `/dev/EtherCAT0` 仅允许 root 访问，因此这台机器使用下面的启动命令。先在当前终端加载上文的 ROS 和新消息包环境；不要 source 旧控制程序的 package，以免加载同名旧 SDK 或旧消息。

```bash
# 启动后即进入 STILL 阻尼，以下不是只读检查命令。
sudo env AIQ11_PRO_HARDWARE_CONFIG="$AIQ11_PRO_HARDWARE_CONFIG" \
  LD_LIBRARY_PATH="$PWD/sdk/lib:$LD_LIBRARY_PATH" \
  AMENT_PREFIX_PATH="$AMENT_PREFIX_PATH" \
  "$PWD/build/aiq11_hw/apps/eir_ctrl_demo_aiq11/eir_ctrl_demo_aiq11" \
  "$PWD/apps/eir_ctrl_demo_aiq11/config/pro_hardware.yaml"
```

`--check-config` 检查配置、策略资源契约和机构资源，并列出 32 项槽位对应关系；它不验证总线在线状态、设备权限或实际机构响应。应用独占 SDK 通信，不同时运行旧 `ehr_bridge` 或其他电机控制程序。

另一终端加载相同 ROS/消息环境，查询并切换模式：

```bash
source /opt/ros/humble/setup.bash
source install/aiq11_hw_interfaces/setup.bash

ros2 service call /mode_switch ehr_message/srv/LocomotionModeSwitch '{cmd: 0}'
ros2 service call /mode_switch ehr_message/srv/LocomotionModeSwitch '{cmd: 2}'
ros2 service call /mode_switch ehr_message/srv/LocomotionModeSwitch '{cmd: 0}'

# 上一条查询返回 ready=true、fault=0 后进入 AMP WALK。
ros2 service call /mode_switch ehr_message/srv/LocomotionModeSwitch '{cmd: 28}'

# 回到 STILL 阻尼；完整退出在控制器启动终端按 Ctrl+C。
ros2 service call /mode_switch ehr_message/srv/LocomotionModeSwitch '{cmd: 1}'
```

MIMIC A–E 为 30–34，沿用 READY 到位及策略切换规则。STILL 是阻尼命令，不能当作断电或只读采集。故障锁定后模式服务不会解除锁定；排障后结束进程并重新启动。

仓库默认 `remote.source: external`，接收原有速度话题及模式服务，不自动启动 GUI。使用原机 SBUS 遥控器时，将 `config/pro_hardware.yaml` 的 `remote.source` 改为 `sbus`，重启应用后生效；本次上传的二进制运行包已启用 SBUS。

遥控映射沿用这台机器的旧桥接配置：Y1 控制前后（通道 2），X1 控制左右（通道 3，反向），X2 控制转向（通道 0，反向）。通道编号从 0 开始，速度上限仍为应用配置中的 1.4 m/s、0.6 m/s 和 1.8 rad/s。

| 遥控器操作 | 模式 |
| --- | --- |
| G 负档 + B 键 | READY 准备站姿 |
| G 负档 + A 键 | AMP WALK；须先 READY 到位 |
| G 正档 + H 正档 | STILL 阻尼 |

正档对应通道值 1722，负档对应 282；未核对实物拨杆的上下方向。按键需松开再按才能重新触发；STILL 组合保持时会持续覆盖其他模式请求。当前未配置 MIMIC 遥控快捷键，仍可通过模式服务切换。

有效 SBUS 输入优先于外部速度指令。SDK 未提供独立的 SBUS 接收时间戳或失联标志，无法保证遥控器掉线自动停机；STILL 是阻尼，并非断电。本轮只核对配置和软件解析，未启动控制器验证实物遥控链路。

## 启动与 STILL 阻尼

手动检查反馈方向时使用独立的 [零增益观察入口与本地 RViz](AIQ11PRO关节方向观察.md)。

`runtime.damping_kd` 为默认值 5.0；`runtime.damping_kd_overrides` 按电机名称覆盖。PRO 硬件配置将左右臂末端各三个电机和颈部三个电机设为 0.5：

| 电机名称 | SDK 逻辑下标 | 启动/STILL Kd |
| --- | --- | --- |
| l_arm_5、l_arm_51、l_arm_7 | 19、20、21 | 0.5 |
| r_arm_5、r_arm_51、r_arm_7 | 26、27、28 | 0.5 |
| neck_1、neck_11、neck_13 | 29、30、31 | 0.5 |
| 其他身体电机 | 其余 23 槽 | 5.0 |

`arm_51` 是第六个手臂电机的名称，不使用解剖关节名 `arm_6` 填写覆盖表。未知/重复名称、非有限值及小于 0.01 的值在配置加载时拒绝。无覆盖项的旧配置继续使用统一阻尼。

该设置用于初始 IMU 等待期间及正常 STILL 的实际 SDK 命令和诊断数据。READY、行走/MIMIC 仍使用各自增益；SDK 急停接口保持原有行为。使用修改后的 runtime 与应用一起编译，部署后重启生效。`--check-config` 可在不初始化硬件的情况下列出各电机阻尼。

## 启动 IMU 等待与诊断

首帧电机反馈可能早于 IMU 数据。应用在 runtime 启动后，保持阻尼并按 `runtime.startup_timeout`（默认 2 秒）等待有效基座 IMU；仅处理新序列反馈，有效之前不启动控制或推理线程。不使用单位四元数替代缺失数据，也不放宽 IMU 有效性检查。

超过期限仍无有效 IMU 时退出，错误包含 `samples`、`imu_count`，有 IMU 槽时还包含 `quat(wxyz)` 和 `gyro`。反馈超时、电机故障、编码器或机构映射错误仍立即终止；运行中 IMU 失效仍锁定故障，不进入启动重试。该修改修复首帧时序问题，不能据此证明实物 IMU 通信正常。

## 映射与线程边界

| 应用电机名称 | YAML 物理槽位 | SDK 收发数组逻辑下标 |
| --- | --- | --- |
| 左腿 1、2、3、4、41、43 | 17–22 | 0–5 |
| 右腿 1、2、3、4、41、43 | 23–28 | 6–11 |
| torso_1、torso_31、torso_33 | 14–16 | 12–14 |
| 左臂 1、2、3、4、5、51、7 | 0–6 | 15–21 |
| 右臂 1、2、3、4、5、51、7 | 7–13 | 22–28 |
| neck_1、neck_11、neck_13 | 29–31 | 29–31 |

YAML 中的 `motor_slots` 声明物理接线；加载时核对 SDK 的 `JointMapping` 名称别名、32 槽完整性、板卡归属、重复槽、CAN 地址、方向、传动比、有限零偏以及编码器身份，再转换为 runtime 所需的 SDK 逻辑下标，不依赖策略数组顺序。SDK 已在底层完成物理/逻辑重排，不能直接用物理槽位访问 `RobotState` 或 `RobotCmd`；`--check-config` 输出的是转换后的逻辑下标。腰部 SDK 编码器槽 0 对应 torso_3（pitch），槽 1 对应 torso_2（roll）；接受 SDK 历史名称 `wasit_r` 及更正后的 `waist_r`。

runtime 只传输电机坐标，应用在启动阶段和普通 IO 线程调用 SDK 的踝、腰、颈正解与速度变换。腰部沿用旧 PRO 控制器的电机模式：输入 `[torso_33, torso_31]`（SDK 逻辑下标 `[14, 13]`），输出 `[torso_3, torso_2]`（俯仰、横滚）。独立腰部编码器保留完整性、有限值和状态检查，原始值保留在 SDK 状态中，不覆盖解算得到的关节反馈，也不因某一路读数为零自动切换来源。控制线程仍只读写固定容量快照；ROS、解算、推理、文件 IO 不进入控制线程。SDK 的方向、减速比、零偏已经体现在输出轴反馈与命令中，应用不重复转换。

原工程腕部 `arm_6 ↔ arm_51` 的 1:1 契约保留。并联关节与腰部没有独立关节力矩测量，`debug/joint.tau_act` 对应项置零表示未提供，不表示真实零力矩；实际电机力矩保留在 `debug/motor`。解算回退、越界、非有限值、编码器缺失或异常状态会锁定故障，不把缓存结果包装成新的有效关节反馈。

`Runner::start()` 失败会立即清理后端；`stop()` 在等待工作线程前请求急停，主程序在等待 GUI 子进程退出前停止控制。保留 runtime 原有急停锁定、反馈/命令超时、手指拒写和迟到命令抑制。

## 本轮软件验证

`ctrl_hardware_test` 只使用注入后端，不创建真实 SDK 实例。覆盖配置错误和机型拒绝、32 槽读写、腰部电机正解及速度（含俯仰编码器为零、两轴非零正负角度和解算失败后拒绝缓存反馈）、SDK 正解、非法反馈、启动失败立即清理、READY→AMP→MIMIC→STILL、反馈断流后恢复仍锁定和正常退出。测试临时 SDK YAML 是软件夹具，不是部署用标定。

2026-10-10 索引复核修正：旧应用直接使用物理槽位访问 SDK 逻辑数组，造成 29/32 路电机反馈、目标及增益错配。旧夹具也按配置槽位生成数据，未检出此问题。现按独立的 SDK 逻辑顺序注入反馈并检查全部 32 路命令；索引、末端阻尼、收发三项用例在旧实现上均失败，修复后通过。开发机 `runtime_test`、`ctrl_hardware_test`、`ctrl_diagnostics_test`、`runtime_simulation_test`、`ctrl_test`、`ctrl_pro_simulation_test` 共六项 CTest 通过。ROS 诊断测试首次因未加载消息包环境失败，加载环境后在隔离域重测通过。SDK 源码和库均未修改；实物关节方向和运动仍未验证。

```bash
cmake --build build/aiq11_hw --parallel 3 --target ctrl_hardware_test runtime_test
ctest --test-dir build/aiq11_hw \
  -R '^(ctrl_hardware_test|runtime_test)$' --output-on-failure
```

完整构建与回归结果见工作空间 `doc/AIQ/AIQ11PRO硬件接入记录.md`。本轮未验证真机运动、地面性能、硬件实际读写权限、真实遥控器断连或真实相机；这些不由应用接线测试替代。
