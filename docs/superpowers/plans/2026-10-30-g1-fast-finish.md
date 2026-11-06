# G1 Fast Finish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 实际快速回站并持续站稳，正常结束保留站立控制。
**Architecture:** 新只读finish reference builder、200Hz末端统计、显式opt-in DDS terminal command decision；原控制模型/主动作不改。
**Tech Stack:** 现有NumPy/PyTorch/MuJoCo/SDK2。
**Spec:** docs/superpowers/specs/2026-10-30-g1-fast-finish-design.md

## Global Constraints
原主动作pose误差<=2e-5；关节限位1e-5容差；floor lift<=.02m/exitsole>=-.002m；reference末姿为default；输出不覆盖；只仿真不真机；正常kp保持/故障damping/watchdog不改；fullgoal仍需原全局验收。

## Review Focus
- 新future goal一致但不能悄悄删主动作或reset物理状态。
- 快速回正是actual0.5s稳定段起点<=1.5s，末10s完整，不能将指令时长当实测。
- qpos顺序/索引及四元数、地面修正和joint限位全部正确。
- 正常收尾target用已施加策略target，不能瞬时改成actual_q或卸力；故障仍damping。
- 不将proprio静态检查称作接触证明；批量关闭与持续控制的边界写清。

### Task 1: Builder and measured finish gates
**Interfaces:** build_fast_finish(source,scene,out,*,exit_seconds=1.,hold_seconds=12.); summarize_finish(metadata,qpos,dq,rootv,t,*,max_recovery_seconds=1.5,required_hold_seconds=10.)。
- [x] RED缺API/前缀保留/不同N/真standhold/invalidduration/nooverwrite/截断与慢回稳不通过。
- [x] 实现/测试GREEN，实际参数.8/1/1.2两源完整物理实验，按结果选最快稳态。

### Task 2: Safe normal final command
**Interfaces:** deploy.finish.completion_command(meta,state,last_target,*,completed)返回target,damping,assessment；metadata.sequence.fast_finish optin。
- [x] RED正常保留lasttarget/kp、fault/legacy/invalidsensor仍damping；实现GREEN。
- [x] DSH机械接dds_sim.finally，仅新profile正常stand保留kp，报告terminal_control；异常末态exitfailure不能被summary覆盖。主代理完整SDK验证。

### Task 3: Full demonstration and review
- [x] native/noise+mass+friction+delay、多次DDS；实测回稳和10s保持，无隐藏reset/轨迹滤波。
- [x] fresh review，一次RED→GREEN重要问题修复，全常规+必要真实检查，生成50fps实际末段视频。

结论：仅采用当前演示来源的0.8s退出作为结束演示。修复后版本三次DDS回稳1.125/1.140/1.120s，名义原生1.12s、两组组合扰动1.185/1.190s；全部末10s稳定，正常standing_hold、断联故障damping。全常规334pass/2skip/9slow deselect，Ruff和格式检查通过。50fps691帧真实末段录像已解码验证；选择/哈希/限制见finetune_projects/g1_continuity_20261029/FAST_FINISH_SELECTION.json。旧保留策略快速退出未全通过扰动，不全局采纳；全段连续性目标继续active。
