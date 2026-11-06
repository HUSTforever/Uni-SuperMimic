# Spectral PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 将实际200Hz高频运动成本纳入PPO并支持难片段采样，保持完整验收要求。
**Architecture:** 独立HighpassBank和SpectralCollector，默认路径不变；PPO仅显式启用新成本。
**Tech Stack:** NumPy/PyTorch/原生MuJoCo，无新依赖。
**Spec:** docs/superpowers/specs/2026-10-30-g1-spectral-ppo-design.md

## Global Constraints
不修改物理推进/PD/参考/策略结构/冻结归一化；不滤波输出/视频；默认成本0/采样不变；一控制步4个实际子步；原始全阶段频谱、风格、DDS、扰动和快速站立门槛不变。

## Review Focus
reset前后成本守恒；逐行filter状态隔离；线性初态初始化；延迟分支5+15ms样本无重复/漏失；无效/超界focus在输出发布前失败。

### Task1: Actual spectral reward
**Files:** Create finetune/spectral.py, tests/finetune/test_spectral.py；Modify finetune/collector.py仅simulator工厂。
**Interfaces:** HighpassBank(num_envs,channels=5,cutoff_hz=8.,sample_hz=200.), reset(row,values,velocity=None), observe(row,values)->filtered；SpectralCollector(bundle,scene,*,ankle_hf_weight=0.,root_hf_weight=0.,**basekwargs)。
- [ ] 写常量/线性/解析正弦增益/逐行隔离/invalid tests，RED→GREEN。
- [ ] 实际collector多次reset和delay=0/1状态逐张量与baseline相同、4samples、奖励和episode成本守恒；RED→GREEN。

### Task2: PPO opt-in
**Files:** physical_ppo.py及test_physical_ppo.py。DSH机械局部接线，主代理验证。
- [ ] 配置权重有限非负、focus整数有效/default禁用、collector正确路由与source不变；RED→GREEN。
- [ ] 完整常规suite、fresh review，一轮Important修复并复验，不反复审查。

### Task3: Evidence and controlled trials
- [ ] 离线原始200Hz轨迹重放和成本/频谱关系审计，记录真实初态/权重/采样配置。
- [ ] 同源同focus两组各200轮、每25保存，全部候选完整原生/风格评估。
- [ ] 合格则验证完整快速回站/源引擎/DDS多次/扰动/真实视频；未合格保持global goal active并记录下一步证据。
