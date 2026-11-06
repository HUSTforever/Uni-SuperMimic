# G1 Balance Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 改善完整舞蹈中的支撑脚滑动和机身晃动，并实际追加训练验证。
**Architecture:** 可选 DeploymentMDP 稳定代价，真实 MuJoCo collector 与 imagined policy reward 共用；保持模型/动作/观测契约。
**Tech Stack:** PyTorch、MuJoCo、现有本地 DDS SDK。
**Spec:** docs/superpowers/specs/2026-10-29-g1-balance-finetune-design.md

## Global Constraints
不覆盖旧产物；reference 不变；hardware_ready=false；不增加在线特权输入。用户已授权持续优化和训练，直接实施。继续现有 codex 分支以保留尚未提交的依赖；不修改其他会话的 live_viewer/log。

## Review Focus
- 默认关闭应与原奖励一致。
- 倾身/转身不应被绝对直立约束抹掉。
- 参考的摆动脚/快速转脚不应被当成固定支撑。
- 代价必须同时进入真实 collector 和可微 imagined rollout，GPU 拷贝正确。
- 新指标不可把几何接近地面当作接触力证明；不能选只改善末尾的模型。

### Task 1: Optional stability reward and focused starts
Files: collector.py, train.py, new balance.py, tests/finetune/test_balance.py.
- [x] 写静止零代价、倾斜/角速度/滑移扰动梯度、默认奖励兼容、非法权重/区间及采样测试，先看到失败。
- [x] 实现默认 balance_weight=0；训练 CLI 加 --balance-weight 和可重复 --focus-interval START END；配置记录精确参数。
- [x] 跑相关测试；主代理验收。

### Task 2: Train and measure
Files: new balance_metrics.py/test_balance_metrics.py, project experiment records.
- [x] 测试静止脚、移动脚和 tilt-error 运动学指标；实现以实际时间戳差分和官方模型映射复算。
- [x] run_003 2000 iterations，100 warmup，500 间隔保存；源 last.pt 的 pi/wm 与 600 checkpoint 一致，hash 保留。
- [x] 所有候选完整 native，动态指标与旧模型对照；候选不达标不能替换旧选择。

### Task 3: DDS and delivery
- [x] 最优候选 3 次本地 DDS、5ms/组合扰动、完整视频与关键片段前后对比。
- [x] 独立代码评审、测试/lint/视频完整解码、hash 检查。
- [x] 新 README/RESULTS/训练方法记录，保留历史；无 Git 身份时仅暂存不提交。

实际新增run_004 1000轮以验证审查修正；最终按统一指标选run_003/2000，未用更多轮数替代质量筛选。验证见项目BALANCE_RESULTS与BALANCE_VERIFICATION。
