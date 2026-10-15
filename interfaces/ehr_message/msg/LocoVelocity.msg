# 足式速度及部分步态参数（话题：/loco_velocity_cmd）

uint8 PRIORITY_REMOTE=0
uint8 PRIORITY_NORMAL=1
# 数值越小优先级越高；默认普通来源，遥控器发布端必须显式设置为 PRIORITY_REMOTE。
uint8 priority 1

float32 linear_vx   # 前后方向线速度 (m/s)
float32 linear_vy   # 左右方向线速度 (m/s)
float32 angular_wz  # 绕竖直轴角速度 (rad/s)
# 摆动足抬腿高度 (m)；>0 时写入 leg_pm.swing_height，锁存至下一次 >0 的字段值
float32 swing_height
# 名义步态周期上界 (s)；>0 时映射到 comd_pm.peroid，锁存至下一次 >0 的字段值
float32 gait_time
# 速度指令保持时间 (s)；超时后 vx/vy/wz 按控制器逻辑归零（与 swing_height、gait_time 锁存无关）
float32 keep_time
# true：行走模式下禁止自动切回站立（与 keep_time 无关，由最后一次话题值锁存至下一条消息）
bool keep_troting
