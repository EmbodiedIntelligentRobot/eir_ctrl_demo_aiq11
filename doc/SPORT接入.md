# SPORT 接入

普通配置启用 **STILL、READY、AMP HUMANOID WALK**；特殊动作由 `config/sport_special_mimic.yaml` 单独启用（35/36 躺下、37/38 起身、42/43 下蹲/起立），新资源尚待闭环动态验收。已完成软件回归与双后端自动仿真检查，详细结果及未通过项见工作空间 [公开版验证范围](../../../doc/公共/验证记录.md)；尚未完成发布验收。PRO 配置、模型和策略入口保持原状。

2026-10-14：新增 A–E 舞蹈资源与独立 `config/sport_mimic.yaml`（30–34，544→23，50 Hz），沿用同一个 MIMIC 实现和推理线程；源参数及测试边界见 [公开版使用说明](../../../doc/AIQ/仿真使用.md)。普通 AMP 和硬件配置保持原启用集合。

## 资源与接口

| 项目 | SPORT |
| --- | --- |
| 运行入口 | `config/sport.yaml` |
| 机器人配置 | `config/robots/sport.yaml` |
| 策略包 | `policies/amp/sport/walk/` |
| 原始权重 | `policy_lhb_0717_4.pt`，保留为 `policy.pt` |
| 模型接口 | float32 `[1,800] → [1,23]`，无循环状态 |
| 关节顺序 | 左腿 6、右腿 6、腰部 3、左臂 4、右臂 4 |
| 机构映射 | 双踝 + 腰部；无 PRO 腕部、颈部和手指 |
| 频率 | 控制/SDK 交换 500 Hz；AMP 100 Hz、特殊 MIMIC 50 Hz；ROS 50 Hz |
| 后端 | `inference.backend: onnx` 或 `libtorch`，不自动回退 |

单帧 80 维：`omega(3), gravity(3), command(3), q-default(23), dq(23), previous_action(23), swing_ratio(2)`。
10 帧按帧展开，由旧到新。首次用静止相位、零速度指令和零上一动作填充历史；进入新会话时重置历史、速度滤波、指令平滑和相位。当前 PRO WALK 为 780 维、50 Hz，没有 SPORT 的相位输入。

`amp_gait` 配置分项裁剪、10 Hz 关节速度低通、速度指令加减速和步态相位。滤波先于裁剪，从零状态初始化；与参考配置的 `fil_gain: [10,100]` 一致。模型输出裁剪为 ±20 后按各关节缩放叠加默认姿态。`source.yaml` 仅作来源追溯；运行只读取 `config.yaml` 与 `model_contract.yaml`。

输出保持参考 `HYBRID_MODE`：膝/踝/腰使用关节侧前馈力矩，阻尼只施加在电机侧（膝 5、踝 3、腰 yaw 3 / roll-pitch 5），避免重复计入。READY 使用参考 `aiq11_control_param.pm` 的目标和增益。角度告警范围来自随包 SPORT MJCF；有限超限仍只告警。前馈力矩上限不高于原应用的分部限值及 MJCF 的 `abs(gear × ctrlrange)`。

## 与参考工程的差异

- 共用既有 AMP 实现和推理线程，不增加线程；PRO 不启用 `amp_gait`。按用户确认，控制路线不引入 WBC、MPC 及其依赖，后续也不迁移相关控制功能。
- 起步脚按侧移方向优先，其次旋转方向，否则使用 `default_left`，默认左脚；须在双向启停、扰动和停止后再起步中验收。参考工程的 WBC 质心/足端选脚不在迁移范围内；必要的并联机构解算继续调用 SDK 库。
- 原 `extra_cycles_on_stop` 实际按接近双支撑的采样次数扣减。本配置命名为 `extra_supports`，保留原计数行为，不宣称是完整步态周期。
- 特殊 MIMIC 使用 `policies/mimic/sport/` 的 544→23 模型、50 Hz；爬起默认不用手，可改为 `getup_prone_original`。感知、motion_stream 和手指尚未接入 SPORT；不得套用 PRO 资源。
- 硬件入口 `config/sport_hardware.yaml` 已接入；23 轴显式映射到 SDK 32 槽位，默认 STILL 锁定，实机尚未验证。配置、构建和首测步骤见 [公开版验证范围](../../../doc/公共/验证记录.md)。

## 后续启动

先按 [应用 README](../README.md) 编译。在工作空间根目录、已加载 ROS 与消息包环境的终端运行：

```bash
/tmp/eir-ctrl-build/apps/eir_ctrl_demo_aiq11/eir_ctrl_demo_aiq11 \
  apps/eir_ctrl_demo_aiq11/config/sport.yaml
```

`runtime.model` 默认相对配置指向 `sdk/share/eir_mujoco/aiq11_sport/model/urdf/aiq11.xml`，不需要设置 PRO 环境变量。模型名必须为 `aiq11`，关节/执行器也须匹配；错机型在 GUI 和物理线程启动前拒绝。修改 runtime 公共头后需重编 runtime 及使用它的应用，勿混用旧二进制。

默认打开 MuJoCo 和遥控 GUI，机器人初始悬吊；仍按 STILL → READY 到位 → AMP（28），READY 到位自动释放。首次动态验收应先将 `runtime.release_after_ready` 设为 `false` 做悬吊检查，再恢复自动释放。QUERY=0、STILL=1、READY=2、AMP=28；RL→READY 禁止，未配置策略拒绝。Ctrl+C 退出并清理本应用窗口。

运行与迁移边界见工作空间 [公开版验证范围](../../../doc/公共/验证记录.md)。自动测试使用无 GUI 仿真，不能替代人工动作观察或实机验收。

## 替换模型

PT、ONNX、默认姿态、关节顺序、动作缩放、历史和相位配置必须成套核对。当前 ONNX 使用 PyTorch 2.11.0 的 TorchScript 导出器、opset 17，模型结构为 800→512→256→128→23。导出方式：

```python
import torch
model = torch.jit.load("policy.pt", map_location="cpu").float().eval()
with torch.inference_mode():
    torch.onnx.export(model, torch.zeros(1, 800), "policy.onnx",
                      input_names=["obs"], output_names=["action"],
                      opset_version=17, dynamo=False)
```

导出后执行 ONNX checker 和双后端数值对照，再更新契约及 `资源校验.json`。只修改哈希不能证明新模型与配置兼容；若模型不再需要相位或频率发生变化，应明确更新观测实现和契约。
