# DebugMotorArray.msg
std_msgs/Header header

uint32 motor_num

# 电机反馈状态（原始硬件数据，未经estimator处理）
float32[] q_act   # rad，按机器人 motor_names 顺序
float32[] v_act   # rad/s
float32[] tau_act # Nm，后端反馈，不保证是独立力矩传感器实测
float32[] current

# 电机指令（发送到电机的命令）
float32[] q_des
float32[] v_des
float32[] tau_des # PD + tau_ff 估算 Nm，不是独立下发字段/限幅后实测力矩
float32[] tau_ff  # 实际下发前馈 Nm
float32[] kp      # Nm/rad
float32[] kd      # Nm*s/rad

# 电机诊断信息
float32[] temperature
uint8[] status_code
uint8[] error_code
