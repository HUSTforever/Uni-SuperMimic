# Projected Residual Implementation Plan

> REQUIRED: superpowers:executing-plans, test-driven-development, verification-before-completion. Inline execution already authorized. Math/architecture/final verification Codex; mechanical factory/PPO wiring dsh.

**Spec:** docs/superpowers/specs/2026-11-02-g1-projected-residual-design.md
**Goal:** 恢复有效的物理动作探索并只学习小幅修正，保持零初始化行为与完整目标。
**Architecture:** 冻结原Policy + outward-rounded raw projection + zero-head residual；显式checkpoint模型类型。

## Constraints / Review Focus

不得改变旧Policy数值路径、参考、机械/力矩限位或PD；delta0全段真实物理一致后才训练。投影仅包base，不包base+delta，否则梯度再次消失。源args/weights不变；base冻结/无重复包装；新增类型的TS与checkpoint严格一致；默认关闭不改变旧路径；未知类型、残差降级和不匹配bounds拒绝。

### Task1: 核心模型

- [ ] Create rl/algos/supermimic/projected_policy.py：PROJECTED_KIND、ProjectedResidualPolicy(base,raw_lower,raw_upper,hidden=128)。暴露base/residual、obs_norm/goal_norm/hist/action_smoothing/num_joints属性。
- [x] 初始11项RED；核心6项GREEN，零残差机械命令身份、有效梯度、base冻结、非法bounds及TS往返已验证。
- [x] Codex实现独立模型类，旧Policy不修改。

### Task2: 明确迁移/导出/训练

- [x] 真实collector、预热、仅残差更新、源/args不变、save/reload/export及降级/bounds校验GREEN；初始相关49通过。
- [x] dsh仅连接指定两文件，主代理检查实际diff。
- [x] 一次Astra审查无Critical，Important的loaded buffer覆盖问题已3项RED→GREEN修复；常规400通过、2跳过、9 slow未运行，Ruff4文件通过。无Minor。

### Task3: 初始完整身份与有限试验

- [x] 实际012125零残差已导出并完整native复跑，50Hz/200Hz qpos、targets、速度与时钟全项最大差严格0；PROJECTED_RESIDUAL_INITIAL_PARITY.json。
- [ ] CUDA短验证通过（仅残差更新，base/WM/norm不变），run020有界500轮已启动；等待18个actor保存点完整评估。环境/参考/门槛不变，不假装探索轨迹与旧算法相同。
- [ ] 合格者走全部源引擎/DDS/扰动/真实视频；无合格继续全目标，不提升默认或声明真机可用。
