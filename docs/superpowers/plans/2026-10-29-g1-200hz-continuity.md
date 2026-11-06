# G1 200Hz Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 针对真实高频颠簸建立全流程验收并改善策略，完整目标保持active直到被验证。
**Architecture:** 默认关闭的纯记录observer、Numpy频谱指标、DeploymentMDP可选踝部指令/速度成本；不改变原始物理推进和Policy图结构。
**Tech Stack:** NumPy/PyTorch/MuJoCo/既有SDK DDS。
**Spec:** docs/superpowers/specs/2026-10-29-g1-200hz-continuity-design.md

## Global Constraints
不覆盖旧模型/录像；不修改参考主体来规避动作；无钉根/隐藏reset/视频轨迹滤波；模拟器和策略契约默认兼容；hardware_ready=false。继续现有codex分支，保留全部未提交依赖与其他会话文件。

## Review Focus
- 200Hz记录不能改变物理或用陈旧传感器构造状态。
- 频谱指标不能用平均值掩盖局部/首尾抖动。
- 新成本须同时用于真实和想象奖励，正确使用实际电机scale。
- MDP.to后的常量/权重在GPU正确；默认关闭保留旧值。
- 训练/source引擎与DDS/end-to-end都验证，不用单侧传输通过代替完整目标。

### Task 1: Reliable high-rate recording and gates
- [x] RED测试：observer每20ms实际记录4个子步，复制值可修改而不改变物理，reset记录真实初态。
- [x] NativeSimulator增加可选observer，evaluate_sequence透传；原默认和物理推进不改。
- [x] RED→GREEN测试：2Hz主运动与10Hz颠簸可区分、1段局部颠簸使最差滑窗失败、首尾都覆盖、采样不足/时钟异常拒绝。
- [x] continuity_metrics.py及continuity_eval.py保存200Hz原始数据和独立严格通过标志，不改旧通过标志含义。

### Task 2: Optional ankle command/velocity cost
- [x] RED→GREEN：静止/参考速率无额外成本；ankle指令交替/实际速度偏差有成本和有限非零梯度；非ankle动作不误罚；默认兼容；非法权重拒绝。
- [x] continuity.py helper，collector.py MDPreward可选成本及scales tuple，train.py CLI/config；不改SuperMimic核心。
- [x] CPU真实短训练与实际CPU/CUDA一致性；适当独立review和全量测试。

### Task 3: Train, inspect, iterate
- [x] 新项目run001按规格1000轮/250保存，从现完整checkpoint热启动；原hash保留。
- [ ] 所有候选完整200Hz物理、阶段/最差窗口/保真/接触比较；不为通过放宽门槛。
- [ ] 选中候选多次DDS、组合扰动、原训练引擎与目标引擎全程验收，生成真实视频及最差片段。
- [ ] 必要时继续训练、控制器或参考可行性优化，目标未证实时保持active；仅完整审计满足后update_goal complete。
