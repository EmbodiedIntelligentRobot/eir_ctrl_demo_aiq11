# DebugJointArray.msg
std_msgs/Header header

uint32 joint_num

# 身体按 joint_names 顺序：位置 rad，速度 rad/s，力矩 Nm。
# 手部复用此类型：位置为 [0,1] 闭合量，速度数组留空。
# 身体 tau_des 当前固定为 0；tau_des_fake 为关节 PD 估算。

float32[] q_act
float32[] q_des
float32[] v_act
float32[] v_des
float32[] tau_act
float32[] tau_des
float32[] tau_des_fake



