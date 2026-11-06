# G1 Contact Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 提高接触速度模型预测质量，再验证是否能改善完整物理策略的连贯性。
**Architecture:** 仅DeploymentLearner可选加权WM误差，默认原样；固定Pi/公共buffer/同种子对照，再做受同一物理门槛约束的策略训练。
**Tech Stack:** 既有PyTorch/MuJoCo/MJWarp/SDK2。
**Spec:** docs/superpowers/specs/2026-10-30-g1-contact-model-design.md

## Global Constraints
沿现codex分支与已存未提交依赖，不覆盖旧数据/选择；不改策略输入输出或PD；权重默认1且有限>=1；保留mask和未加权wm_err8；只有完整目标被证明才可完成goal，hardware_ready=false。

## Review Focus
- 4踝qd索引不能误作q；根速度权重覆盖局部3维，不能误称世界竖直轴。
- mask后的状态不能进入loss/梯度；last_error不随新权重改变口径。
- 默认必须数值原样；非法权重在目录/模型构建前拒绝。
- CPU/CUDA/compile路径上的权重常量不能留错设备。
- 固定Pi两分支公共数据和norm不变，heldout不能被训练，模型变好不等于物理策略变好。

### Task 1: Weighted deployment WM error
**Files:** finetune/learner.py；tests/finetune/test_wm_weighting.py。
**Interfaces:** DeploymentLearner(...,wm_ankle_velocity_weight=1.,wm_root_velocity_weight=1.)；_wm_err输入/返回与基类相同。
- [ ] RED：真实State误差测试四踝qd与根线速度的loss比例、非目标维度不变、mask梯度为零、last_error口径不变、非法权重拒绝、默认与基类逐值一致。
- [ ] 实现仅部署子类override；运行新测试和现learner测试，Expected全部PASS。

### Task 2: Training flags and fair WM-only experiment
**Files:** finetune/train.py；tests/finetune/test_training.py；project下contact_wm_fit.py与诊断缓存脚本。
**Interfaces:** run_training两个weight关键字/CLI/config；WM-only脚本复用Task1，输出完整WM检查点、Pi不变证明与对照元数据。
- [ ] RED真实CPU短训练传4/2，配置保存且非法输入不建目录；GREEN机械参数透传（DSH）。
- [ ] 固定公共buffer、源模型/norm/Pi、数据hash、同batch种子，600更新/分支，报告独立heldout多步物理误差。
- [ ] 主代理读diff、复跑测试、实际CPU/CUDA与compile/模型拟合验收，Expected默认兼容且有真实对照结果。

### Task 3: Physical policy validation and iteration
**Interfaces:** 消费Task2通过实测选择的完整checkpoint，不要求加权分支必须胜出。
- [ ] 预测改善后新目录进行策略微调，导出各候选，原五阶段200Hz连续性/保真/稳定筛选。
- [ ] 合格候选原训练MJWarp与独立MuJoCo/DDS、重复及组合扰动和完整真实视频。
- [ ] 一次fresh review/RED→GREEN修重要项、全常规测试与必要真实检查；不能由局部门槛推断完整目标。
