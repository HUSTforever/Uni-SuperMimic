# Integrated Servo Probe Plan

**Spec:** docs/superpowers/specs/2026-11-02-g1-integrated-servo-probe-design.md

实验是既有流水线的有界数据对照，使用现有实现，不修改产品行为。Codex负责假设与最终验收，dsh仅执行复制/固定配置和既有完整仿真命令。不新增镜像实现的单元测试；实际baseline逐状态对齐是必要验证。

1. 复跑fast_integrated_base_v1，核对与已存完整物理qpos差<1e-10；失败停止。
2. 三个新bundle仅改4个ankle kd的1.25/1.5/2倍，保存源/策略/参考hash和全部控制变化；源不动。
3. 逐组运行现有projection_eval.evaluate，再用200Hz summarize_finish，保存完整报告和统一结果。四组都用相同完整门槛。
4. 主代理检查参数/哈希/实际结果与原始报告；有合格候选才推进其完整simtosim，无合格则记录改变下一动作的证据并保留全目标。
