# Physical PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** 用真实物理回报微调原均值策略，减少对接触世界模型偏差的依赖，并沿用全部验收门槛。
**Architecture:** 纯PPO数学函数 + 复用Collector的独立训练器；原Policy/Critic/export保持不变。
**Tech Stack:** PyTorch/NumPy/现有部署MuJoCo。
**Spec:** docs/superpowers/specs/2026-10-30-g1-physical-ppo-design.md

## Global Constraints
不覆盖源；pi结构、固定归一化、PD/参考不改；WM不更新；latentsample logprob不得使用clipped动作；GAE不跨reset；不改完整连续性/保真度/扰动门槛；不连接真机。

## Review Focus
- timeout与失败bootstrap不同，next_value来自重置前状态。
- 新episode第一历史重复且previous_action是实际投影输出。
- source pi与bundle不同必须在创建输出前报错；原权重不变。
- 只冻结归一化不应冻结策略；critic-only warmup不得改pi。
- 所有PPO超参数有限合法，异常与已存在输出不能覆盖。

### Task 1: PPO math and physical rollout
**Create:** src/unisupermimic/finetune/ppo_math.py、physical_ppo.py；tests/finetune/test_physical_ppo.py。
**Interfaces:** generalized_advantages(rewards,values,next_values,terminal,done,*,gamma=.99,lam=.95)->(adv,returns)；clipped_surrogate(log_prob,old_log_prob,advantage,*,clip=.1)->(loss,stats)。PhysicalPPOConfig dataclass；PhysicalPPO(bundle,scene,checkpoint_path,config,*,device='cpu')；collect()->dict of stacked T,N observations/goals/latent/actions/logprob/value/reward/terminal/done/next_value/GAE/return；update(batch,*,actor=True)->finite stats；close()。
- [x] 写GAE手算timeout/terminal与跨reset、clipping正负adv测试；验证RED，完成math后GREEN。
- [x] 实现直接Collector rollout和warmstart，真实collector+sourceTS均值历史目标对照、latentsample和投影的记录边界；RED→GREEN。

### Task 2: Training and original export
**Interface:** run_physical_training(bundle,scene,checkpoint,output,config,*,device='cpu')->report；CLI --bundle/scene/checkpoint/output/iterations/num-envs/rollout-steps/device，其余可选参数由config默认。
- [x] 真实短训练：critic warmup pi严格不变；actor更新非零；WM/obsNorm/goalNorm和source均不变；PPOcheckpoint经原export成功；无效参数/源身份错误/输出存在不发布。RED→GREEN。
- [x] 实现保存初始/周期/最终checkpoint与config/逐轮metrics，记录物理步数，完整保存PPO优化器但不冒充旧优化器。

### Task 3: Controlled physical trial
- [x] 初始确定性实际全段对齐原生运行，报告最大轨迹/目标差；一次fresh review并修复Important，常规suite验证。
- [x] 运行首200轮并每25轮独立评估完整原生物理/原始保真度；不按训练奖励采纳。
- [ ] 依据证据继续迭代或对合格候选验证快速回站、多次DDS/扰动和真实视频；若未通过保留global goal active。

已额外完成仅实际速度项×20的200次更新、8个候选评估；两组共16个完整运行、无全项通过。最佳第二组175轮HF.008126/RMSE.126460，保真通过，HF仍超标；不采纳。下一调查为19.5～27秒重复高频峰值与频段匹配的真实物理训练成本，保持原始参考和完整验收门槛。
