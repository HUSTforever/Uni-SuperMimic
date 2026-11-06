# G1 Ground Slide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 减少参考贴地运动的实际滑动并追加训练。
**Architecture:** 可选可微脚部代价，经collector和imagined rollout共用；只加配置，不改部署接口。
**Tech Stack:** PyTorch/MuJoCo/既有DDS。
**Spec:** docs/superpowers/specs/2026-10-29-g1-ground-slide-design.md

## Global Constraints
参考/物理/观测/动作契约保持；旧产物保留；hardware_ready=false；无硬件操作。

## Review Focus
快速竖直摆动和高脚不能误标；旧默认行为必须保持；新参数必须记录；实际接触诊断不可当作通用稳定证明；部署不输入特权接触力。

### Task 1: Optional reference-ground cost
- [x] tests/finetune/test_ground_slip.py先RED：fast horizontal reference is penalized,vertical/high foot excluded,finite velocity gradients,invalid parameters rejected。
- [x] balance.py实现ground_slip；collector.py/train.py接线ground_slip_weight、配置/CLI默认0；真实短训练验证。
- [x] targeted+normal suite、独立审查。

### Task 2: Actual training and candidate selection
- [x] run006 1500轮，从run003 last起步；原hash保留。
- [x] 500/1000/1500完整native，旧四项/近地/实际承载接触指标对比，按规格筛选。

### Task 3: DDS and artifacts
- [x] 有改善时3次实际DDS、延迟/扰动、真实50fps视频；否则保留旧模型。
- [x] 源hash/归一化/视频解码/测试验证，更新方法和选择记录。

实际完成run005低LR1000、run006强约束1500、run007半剂量1000、run008中间剂量500共4000轮；run009另200轮复现完整热启动状态。最终run008/200通过全部门槛与真实DDS，结果见GROUND_RESULTS。
