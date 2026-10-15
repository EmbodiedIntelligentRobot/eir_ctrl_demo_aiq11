# eir_ctrl_demo_aiq11

基于 eir_loco_runtime 的纯 C++ RL 控制应用，支持 AIQ11 PRO 与 SPORT。ROS 2 负责遥控、模式服务及可选诊断，不使用 ros2_control 或 hardware_interface。

- STILL → READY → RL；AMP、MIMIC 和感知策略分别实现。
- LibTorch / ONNX Runtime 由配置选择，不自动回退。
- PRO 与 SPORT 独立模型、关节映射和策略配置；PRO A–E 提供 fist / gripper 两组资源。
- 推理、ROS 发布与记录写盘不在实时控制线程执行。

本公开分支配合 [workspace](https://github.com/EmbodiedIntelligentRobot/eir_locomotion_workspace/tree/publish) 的 SDK、模型资源和构建入口使用。构建、仿真与验收范围见 workspace 的 doc/公共/ 和 doc/AIQ/。

基础 ONNX 入口为 config/pro.yaml、config/sport.yaml。默认无参数入口为 config/pro_perception_9500.yaml，默认使用 LibTorch；单后端构建需在配置副本中选择可用后端。PRO 六个特殊动作使用 pro_special_mimic.yaml，SPORT 使用 sport_special_mimic.yaml。

接口独立构建说明见 [interfaces](interfaces/README.md)。策略结构与替换见 [配置说明](doc/策略配置与替换.md)。配置中的参数和策略权重保持本次已集成版本，发布分支整理不代表新增实机验收。

自有开放源码采用 [BSD-3-Clause](LICENSE)，第三方接口及资源保留原许可。
