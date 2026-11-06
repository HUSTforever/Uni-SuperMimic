# 真实物理回报的连续性微调

目标仍是从起始到快速回站全段无颠簸，并保留主要动作。上一轮是 progress：快速回站实测通过；舞蹈窗口 HF 与原始风格门槛仍未全通过，不能标记全局完成。

## 证据和选择

现有 DeploymentCollector 使用真实部署 MuJoCo/PD，但 DeploymentLearner.policy_update 的梯度全部经过 WM.step。固定策略审计显示世界模型会低估局部接触速度成本；修正参考和加权拟合尚未解除完整指标失败。机械检索确认仓库没有正在使用的 PPO/logprob 接口。

可继续改进世界模型，代价是策略仍可能利用接触预测误差；也可更换控制器，代价是改变训练/部署契约且过去 PD 对照未充分改善。本轮 Codex 在已授权的持续迭代范围内选择真实物理回报的 PPO 微调。这是实施选择，不称用户指定了算法。算法参考 https://arxiv.org/abs/1707.06347 。

## 训练契约

复用 DeploymentCollector、原始 Policy/Critic、原始参考、限位投影和候选导出，不改默认、仿真步长、PD 或部署网络结构。源为 candidates/run002_000250，完整暖启动 run_003_reproduce/last.pt；其 dance RMSE .12691、最差 HF .00897，仍未合格。启动前验证检查点均值策略与源 TorchScript 一致。归一化固定，完整 checkpoint 的 WM 原样携带，不参与任何策略梯度或更新。

从同一均值策略输出采样固定 std=.03 的高斯 latent raw action；只在环境执行时投影到关节限位。logprob 对应未投影 latent。部署仍使用确定性均值，无额外滤波。真实奖励沿用原 tracking、action rate=-.3、balance=1、ground slip=.35、ankle rate=.05，不在首轮同时修改奖励和参考。训练输入先不添加额外观测噪声，使用环境随机初态、摩擦和0/5ms延迟；扰动验收仍独立执行。

PPO clipping=.1，gamma=.99，lambda=.95，固定 std；保留旧均值策略作为锚，anchor=.1 * mean((mean-anchor_mean)^2/std^2)。策略 LR1e-5，critic LR1e-4，梯度范数<=1，近似KL>.015停止该轮进一步策略更新。先只训练 critic 10轮。默认32环境、32步/轮、4epochs、batch256；先200轮，每25保存，按独立完整物理评估决定后续。所有参数和真实样本数记录；不得把训练loss当通过证据。

首200轮8个候选均未全通过。真实随机采样256个舞蹈步显示踝惩罚的实际速度部分只占约4.83%，仅约0.128%的tracking奖励。下一组受控对照仅把该部分乘20，指令项和其他参数、源检查点、参考/PD全部相同。新增可选measured_velocity_multiplier默认1，有限且>=1；只给PPO的MDP设置ankle_measured_weight=.5*multiplier。旧训练路径缺少该属性时仍严格使用.5，默认奖励不变。新权重是否有效仍按完整原始门槛判定。

GAE 必须使用自动重置之前 s1 的 next_value，真正失败/参考结束不 bootstrap；时间上限 bootstrap 但递推不得跨 done。历史在 ep_step=0 重复真实第一观测，否则正常移位，上一动作必须是实际投影后的动作。保留完整 source args 以复用导出；新 PPO optimizer 单独保存，不能冒充旧世界模型优化器的连续状态。

## 交付和验收

实现有限、可保存的独立训练入口，不修改现有 WM 训练路径。新增 math 边界测试、真实短训练/权重冻结/源身份与导出测试，先 RED 后 GREEN。一次新上下文代码审查，Important 一次修复并全套验证。

固定原门槛：200Hz全阶段/2秒窗口 HF、完整运行、舞蹈RMSE<.13以及原始主要关节幅度>=.8/相关>=.9。任何候选通过后再合并0.8秒快速回站，重新验证最终版本的原生/多次DDS/扰动/10秒保持/50fps真实视频。若本轮没有通过，保留证据和候选，继续完整目标，不替换默认、不连接真机。
