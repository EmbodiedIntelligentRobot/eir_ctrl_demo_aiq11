# ROS 接口依赖

这里提供独立构建所需的 `ehr_message` 与 `prime_msgs` 接口，不是旧工程完整消息包的替代品，不能与同名完整包同时加入同一工作空间。`DebugRuntime` 新增反馈序号、单调时钟时间、年龄、有效标志及控制器故障字段；使用旧版消息的发布端、订阅端和应用必须一起重新构建，不能混用两种消息定义。模式服务原有常量保持不变。

```bash
source /opt/ros/humble/setup.bash
colcon --log-base /tmp/eir-interfaces-log build \
  --base-paths apps/eir_ctrl_demo_aiq11/interfaces \
  --build-base /tmp/eir-interfaces-build --install-base /tmp/eir-interfaces \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source /tmp/eir-interfaces/setup.bash
```

模式服务保留完整常量定义以保持协议兼容；控制器实现 QUERY=0、STILL=1、READY=2、WALK_AMP_HUMANOID=28、MIMIC A–E=30–34。READY 仅接受 STILL 来源；其他未实现模式明确拒绝。

可选上半身离线动作使用原协议 `ehr_message/srv/MotionCmd`：实现 STOP=2 及已配置的 11–255 号离线动作，其他命令明确拒绝。接口文件保留参考工程完整定义；已有消息安装不包含此服务时需重编接口和应用。遥控 GUI 默认调用 cmd=11～21、session_id 为空，通过 result/err_code 显示请求结果；控制应用需使用启用 motion_stream 的配置（例如 pro_motion.yaml）才提供播放服务。
