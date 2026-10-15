# DebugRuntime.msg — 运行时统计
std_msgs/Header header

# 主循环各阶段耗时 (秒)
float64 prime_ctl_duration
float64 ctrl_loop_duration
float64 wbc_solve_duration
float64 mpc_solve_duration
float64 estimator_duration

# 控制器内部耗时分解 (微秒)
string  ctrl_name       # 当前控制器名称
int64   obs_time_us     # 观测计算耗时
int64   infer_time_us   # 最近一次推理计算耗时；非策略模式可能保留上次值
int64   flow_infer_time_us  # Flow encoder 推理耗时 (无 flow 则为 0)
int64   act_time_us     # 动作计算耗时

# header.stamp 是 ROS 发布时间；下列时间使用进程内单调时钟，不能与 ROS 时间相减。
uint64 feedback_sequence
uint64 feedback_stamp_ns  # IO 首次观察到该反馈序号的时间，不是硬件采样时间
int64 feedback_age_ns     # 无反馈或未来时间为 -1
bool feedback_valid      # 有效、未过期且结构完整；否则不发布身体/手部数组
uint64 command_stamp_ns   # 控制命令生成时间；故障后可能保留最后一条
uint8 controller_fault    # 控制器故障枚举，不是电机 error_code
uint32 fault_joint       # 64 表示未定位到单个关节

