# PRO AMP WALK 模型

当前策略：`aiq11pro_female_standing_foot_motion_refine_local_74994`，任务 `aiq11_pro_female_walk`，训练 run `2026-10-12_22-34-39_female_standing_foot_motion_refine_local`。

用户指定 `policy_local_1013_0913_standing_foot_motion_74994.pt`。交付 `policy.pt` 保留其原始 TorchScript；`policy.onnx` 来自同一导出目录，权重逐项与 PT 完全一致。导出参数和来源记录分别保存于 `source.yaml`、`source_metadata.json`。

- CPU float32，输入 `[1,780]`，输出 `[1,23]`，ONNX opset 18；输入名 `obs`，输出名 `actions`。
- 78 维 × 10 帧，旧帧到新帧；每帧依次为 omega、projected gravity、速度指令、关节位置偏差、关节速度、上一动作。
- 23 个关节的顺序、默认姿态、动作缩放、Kp/Kd、四个踝速度屏蔽、速度范围和 50 Hz 推理与原配置一致。归一化关闭，模型内动作裁剪 ±100。
- 导出元数据记录原地动作区间：|vx|≤0.3、|vy|≤0.05、|wz|≤0.05，该区间是训练奖励的原地目标，actor 输入仍为原始指令。应用未新增速度清零或区间门控；低速指令的实际响应需结合该策略验收。
- READY、限值、机构保护、上半身过渡、MJCF 与物理时间步不因模型更换而调整。

| 资源 | SHA256 |
| --- | --- |
| PT | `2c59ca1e9627a9c5a0f6a4cf6cde0a3c87bf1e27612257fdb0c4d7b83f3c1e1f` |
| ONNX | `2e40bfd2ad148f5bf784f0e24b9e0137cc32859e6d1e4c2d1320040024032ea2` |
| 原始 YAML | `328e7be70f48f97726983bf2923cb591515cff0142941d2232acc3292a6ea1b3` |

## 验证与复现

Python 完成 1,201 组双后端对照：零输入、标准正态与大幅值共 801 组采用 atol=1e-4、rtol=1e-4，全部通过；400 组 ±100 裁剪边界压力输入中有 1 组超出该严格容差。边界最大动作差 4.158e-4，折算目标最大差 1.040e-4 rad；边界使用 atol=1e-3、rtol=1e-4 的独立压力检查通过，没有降低正常输入判据或修改保护参数。

C++ 软件回归已通过，包含 LibTorch 2.7.0 CPU 实际加载和 100 组随机输入与 ONNX 的绝对误差 ≤1e-4 对照。完整模型的本轮仿真结果与验收边界见工作空间 [策略更新记录](../../../doc/AIQ/策略更新记录.md)。

交付 ONNX 使用原目录配套文件，不是旧模型的 ONNX。需要独立重新导出时：

```python
import torch
import onnx
model = torch.jit.load('policies/amp/pro/walk/policy.pt', map_location='cpu').eval()
with torch.inference_mode():
    torch.onnx.export(model, torch.zeros(1, 780), 'policy.onnx',
                      input_names=['obs'], output_names=['actions'],
                      opset_version=18, dynamo=False)
onnx.checker.check_model('policy.onnx', full_check=True)
```

重新导出的文件可能具有不同哈希，必须重新核对数值并更新 `model_contract.yaml`。Python 校验环境：PyTorch 2.11.0+cu130（CPU 推理）、ONNX 1.21.0、ONNX Runtime 1.23.2。
