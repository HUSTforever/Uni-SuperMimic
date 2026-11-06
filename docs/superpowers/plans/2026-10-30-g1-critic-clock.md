# Critic Clock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 消除静止尾段有限终止造成的critic时间混淆，策略契约不变。
**Architecture:** 可选ClockCritic + 严格旧权重升维；PPO追加privileged remaining-step feature。
**Tech Stack:** 现有PyTorch。
**Spec:** docs/superpowers/specs/2026-10-30-g1-critic-clock-design.md

## Global Constraints
默认关闭时critic结构和推理路径保持不变；仅训练critic改输入，Pi/WM/归一化/PD/参考不改；质量门槛不改；源文件不覆盖。训练使用learner独立的rollout/update随机数生成器；新旧代码不能仅凭同seed视为配对采样。

## Review Focus
剩余步数使用真实frame（reset前next_frame正确）；gamma1不能静默全零；旧权重升维不丢参数；新clock checkpoint不能无声明降维；常量站立片尾不应被估成巨额负优势。

### Task1: Clock critic and weights
**Create:** finetune/clock.py、tests/finetune/test_critic_clock.py。
**Interfaces:** ClockCritic(in_dim,hidden,gamma), forward(x)读取最后一维remaining/100；with_clock_column(state,in_dim)->newstate，只允许等维或新增一列。
- [x] 解析恒奖回报/终点0/GAE0/gamma非法、旧列保留和新增列零测试，RED→GREEN。

### Task2: PPO optional clock
**Modify:** physical_ppo.py；默认critic_clock=False，CLI自动布尔开关。
- [x] CPU真实collector的warmstart升维、actor不变、原导出、末帧clock0、下一帧取reset前终点、配对初始动作一致，RED→GREEN。
- [x] fresh review一次，无Critical/Important；Minor的reset前终点测试已补充。常规suite 373通过、2跳过、9 slow未运行。
- [x] 实际大模型CUDA短训练2轮、32个物理控制转移完成；914→915迁移原列精确保留，critic预热不改pi/WM，actor更新改pi，WM/策略归一化保持不变。证据CRITIC_CLOCK_SMOKE_VERIFICATION.json；不代表效果达标。

### Task3: Controlled actual trial
- [x] run016/017各500轮完成，唯一配置差为critic_clock；各512000个真实控制转移。18个actor保存点/组完整评估，另以逐张量身份验证复用25/50预热点的源物理证据。CRITIC_CLOCK_TRAINING_VERIFICATION.json与CRITIC_CLOCK_SUMMARY.json。
- [x] 无原生全项合格者，不提升默认，不把训练loss改善当连续性通过；全目标保持active。新候选的源引擎/DDS/扰动/视频门槛仍须后续实际通过。
