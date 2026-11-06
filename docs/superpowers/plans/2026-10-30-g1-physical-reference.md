# G1 Physical Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 在保留原主要动作的前提下，校正参考的接触/支撑要求并验证实际全流程动作。
**Architecture:** 独立参考数据实验；原上肢和节奏不变，低频物理root/下肢有限混合；所有结果双重保真锚定。
**Tech Stack:** 既有NumPy/PyTorch/MuJoCo/SDK。
**Spec:** docs/superpowers/specs/2026-10-30-g1-physical-reference-design.md

## Global Constraints
不改旧产物/默认模型/PD；1799帧phase原样；参考滤波不修改真实运动/视频；新旧参考双方的实际关节RMSE都须<.13；原完整目标门槛保持；无真机通信。

## Review Focus
- 混合只作用root及q0:15，arm15:29必须原样，quaternion用SLERP。
- envelope起停必须连续，穿地不能逐帧max产生尖点。
- 原style锚点固定，不以参考改写来通过保真。
- centroidal估计是诊断，不能冒称物理可行性/无抖动证据。
- bundler前后帧数/速度/goal/hash与PD必须一致。

### Task 1: Prepare constrained physical projections
**Interfaces:** prepare_projection.py输出wxyz qpos NPZ及provenance JSON；pack_motion_bundle.py消费array/source/metadata发布只读新bundle。
- [ ] 生成.5/.75参考；主代理验证arm原样、主要腿部幅度/相关性、帧数/phase/limits/地面/全qRMS约束。
- [ ] 机械bundle包装由DSH，主代理读diff/加载/复核全部数值和hash。
- [ ] 参考支撑估计与旧参考/真实物理对比，实际原生200Hz测试，附原style锚点分数。

### Task 2: Fit policy only when justified
**Interfaces:** 满足Task1风格及几何的bundle，源run003_reproduce完整权重，原style锚点。
- [ ] 新目录微调、每100保存；同时满足两参考保真及原200Hz/稳定门槛才进入更多DDS。
- [ ] 通过者原MJWarp/独立MuJoCoDDS重复与组合扰动、完整原始视频；未全证实时goal继续active。
