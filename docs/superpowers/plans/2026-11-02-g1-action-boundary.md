# Action Boundary Regularizer Implementation Plan

> REQUIRED: superpowers:executing-plans, test-driven-development, verification-before-completion. User authorized inline continuous work; mechanical wiring belongs to dsh, math and final verification to Codex.

**Goal:** 为已裁剪动作的PPO探索死区提供显式训练梯度，不改变部署契约。
**Architecture:** 纯torch boundary loss + 默认关闭的PPO配置/actor loss连接。
**Spec:** docs/superpowers/specs/2026-11-02-g1-action-boundary-design.md

## Global Constraints

策略/critic结构、归一化、源文件、参考、PD、机械限位与全部质量门槛不改。weight0不影响旧计算图和RNG。critic预热不改actor。完整目标仍需全链验证。

## Review Focus

raw范围由每关节scale/default/offset转换；罚项作用于真实均值而非latent或applied；零权重不影响旧路径；预热及早停不意外更新actor；新checkpoint仍可由旧actor导出。

### Task 1: 数学核心与回归

- [ ] Create src/unisupermimic/finetune/action_bounds.py: action_boundary_loss(mean,lower,upper,std)->scalar；mean为[batch,joints]、bounds为[joints]。
- [ ] Create tests/finetune/test_action_bounds.py，先RED解析梯度/合法边界0/梯度步回可行域/无效输入。
- [x] Codex最小实现，数学9项GREEN（初始13项RED含后续配置/集成）。

### Task 2: 默认关闭的PPO集成

- [x] 配置/集成先RED，dsh只作physical_ppo.py的精确连接；Codex检查实际diff。
- [x] 相关39项GREEN；常规386通过、2跳过、9 slow未运行；Ruff/check与format通过。
- [x] fresh Astra审查无Critical/Important；非零正则的KL早停测试覆盖建议为Minor，记录后延。现有代码早停位于正则计算/更新之前。

### Task 3: 固定源实际试验

- [x] 128步实际CUDA配对：初始动作/物理历史/奖励/HF成本/critic输入和值完全一致。CUDA两步短训练，critic预热不改pi，actor确更新且正则loss>0，WM/norm不变。ACTION_BOUNDARY_SMOKE_VERIFICATION.json。
- [x] run019_bound_mean500轮完成（512000控制步、444.70秒），weight1，其余与run018相同。18个actor保存点均无全项通过，不推广；核心正则默认0保留为实验开关。完整目标未完成。
