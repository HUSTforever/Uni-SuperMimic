# 投影基准动作上的残差微调

## 目标与依据

完整目标不变：保留主要动作，全阶段200Hz抗抖、原始动作保真、快速回站、源引擎/DDS/扰动和真实录像都须通过。此前run019均值边界正则500轮/18个候选全部失败，不采用其权重；原生默认和全部旧产物保留。

固定源策略的诊断表明深度越界提议使20000次std=.03扰动仅产生一个裁剪后的关节动作。离线投影原始提议到向外取整一float32 ULP的raw边界，可以逐值保留已有2038×29个最终机械目标，同时恢复有效探索。该证据仅为记录命令身份和采样性质，还没有新网络或物理重放通过。

## Approaches / Codex选择

继续压低旧提议的正则已导致物理退化；单纯straight-through clamp不能保证前向输出脱离旧参数的平坦区。此次选择冻结原策略，在其投影后的真实执行基准外添加零初始化残差网络。用户已授权持续迭代并不重复确认；本次架构与细节由Codex选择，不冒称用户指定。

## 架构与接口

新增ProjectedResidualPolicy，输入仍(hist×99,goal730)，输出仍raw29。base为原Policy，参数冻结、两个RunningNorm冻结，base完整state逐张量保留。基准为clamp(base(obs,goal),raw_lower_out,raw_upper_out)。物理raw边界以float64从可信metadata的limit/default/offset/scale计算，再转换float32并各向区间外nextafter一ULP，确保零残差后控制器原hard clamp产生完全相同的机械目标。

残差特征为原base的归一化历史和goal encoder的tanh latent；128单元ELU隐层及29输出线性层，末层权重/bias初始化0。输出为projected_base + base.action_smoothing*residual(features)。基准网络独立调用原forward，避免改动旧Policy的数值路径；只训练残差，已有critic、WM、归一化、参考与PD均不变。原关节/力矩限幅及失败看门狗保留。该结构不是视频平滑或参考动画。

config projected_residual默认False、严格bool。启用旧checkpoint时迁移为新wrapper，保存独立policy_args（不得改源args），policy_type='projected_residual_v1'，residual_hidden=128，residual_raw_lower/upper保存float64物理raw边界列表。demo.export_policy.build_policy显式支持此类型并严格加载wrapper state；旧类型保持原路径，未知类型拒绝。新source在同模式下直接加载，不二次包装；不得无声明移除残差类型，保存边界必须匹配可信bundle。

source identity验证先于迁移。anchor为新wrapper的冻结副本，因此anchor成本度量物理基准上的修正，而非旧未执行的大提议。PPO只将requires_grad=True的参数交给actor优化器。save覆盖args为新policy_args；export_metadata的外部输入/输出契约不变，TorchScript的全部base、残差和bounds buffers仍接受逐张量身份验证。新checkpoint仅承诺此PPO路径和导出支持，不宣称旧通用SuperMimic训练器可直接恢复。

## 验证门槛

先RED后GREEN：零残差保持实际机械命令（包括越界、非零previous-action）；投影后残差仍有梯度且可立即改变机械目标；base参数/归一化不变；非法bounds拒绝；TorchScript非零残差、不同batch往返一致。真实collector预热不改actor、更新只改残差，save/reload/export与source args不变、禁止模式降级/边界不匹配。

完整常规suite和一次fresh review通过后，先导出真实源012125的零残差bundle，要求完整原生qpos及target与fast_integrated_initial_base逐状态一致（maxabs<1e-10），不通过则不得训练。通过后CUDA短验证，开始有界500轮run020；与run018相同环境/奖励/超参（clock、完整episode2038、HF1/1、LR3e-6等），action_bound_weight=0，仅改变actor参数化。初始随机探索分布会改变，不称为与旧run同噪声的配对轨迹。

全部18个actor保存点均按原门槛验收；只有原生全项合格者才推进源引擎、DDS、正确质量扰动及原始视频。零残差复现旧动作不代表消除抖动；全目标尚未完成。
