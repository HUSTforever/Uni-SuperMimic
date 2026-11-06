# G1快速回站与主动站立保持

用户明确要求动作结束时快速回到站立、防止摔倒。目标是实际回稳，不是仅缩短动画；保持原主动作/控制器/PD/故障看门狗。当前DDS实测舞蹈结束后稳定站姿约2.23s(保留版)/2.22s(实验候选)，原参考exit3s/settle5s。DDSfinally无论成败都发送kp0阻尼，再关闭模拟器；正常完成与故障应区分。

## 取舍
选择重采样原C2退出轨迹到.8/1.0/1.2s并保持原反馈策略，随后12s主动站立参考，验证最后10s稳定。拒绝瞬时强切default_q；暂不引入另一套站立策略或重训整个舞蹈。以上是Codex在用户授权内的实施选择，不重复请求许可。

原0..dance_end的root/q/旋转姿态原样复制(重建误差<=2e-5)，future goal/速度按新时序重算，允许末200ms预备回站。newexit均匀重采样旧exit，SLERP quaternion，最后重复真正的原静态站姿。source必须sequence五阶段、末姿态与default_q匹配、末旋转直立，source及新q都在可信限位内。退出脚底穿地用64u^3(1-u)^3 C2抬升，最大2cm、只作用exit、两端为0，不能修改主动作或钉根。包保存源hash/参数/相同policy与PD、新segments/真实帧数。

新finetune.finish.build_fast_finish(source,scene,output,exit_seconds,hold_seconds)原子发布只读bundle。默认exit1s、hold12s；最终选择最快通过的测试时长，实际连续0.5s满足站姿条件的开始时间须<=1.5s，末10s持续满足。站姿条件每个200Hz真实样本：default_q RMSE<.10rad、全关节qd RMS<.30rad/s、root速度<.10m/s、pelvis tilt<.20rad；必须实际全程完成且末端数据完整。保留旧输出；hardware_ready=false，未完成整个连续性目标。

验收补充：回稳点取最终连续稳定区间的开始，该区间必须一直覆盖到仿真结束并至少持续0.5s，避免短暂安静后再次晃动也被算作回稳。增加root高度>=.35m；末10s仍独立检查。

DDS收尾只对新sequence.fast_finish配置启用：正常完成且最终proprio检查通过时，最后一帧保持最后已执行站立策略target与原kp/kd，不发kp0；故障/未完成/无有效末态仍damping。检查是关节/IMU静态检查，不冒称可证明脚底接触；实际root速度/完整10s保持由模拟器独立评估。批量模拟器仍会正常关闭，不能把最后一条command当成进程关闭后能无限站立；失联看门狗不改。

需要完整native试验与独立DDS回环/5ms延迟和摩擦质量噪声组合验证，出新结束片段真实50fps录像。全局舞蹈抖动/风格未达标仍保持goal active；这项仅能完成结束行为，不能宣称全sim2sim无问题。
