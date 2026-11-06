# G1全流程200Hz连续性优化

## 目标与现状

用户目标保持为：从开始到结束动作连贯、无颠簸，完整sim2sim可靠。现模型run008/200的3次DDS完成记录仍有效，但不能据此认定无颠簸。200Hz复查发现踝部9–10Hz振荡；踝pitch指令>8Hz RMS约.010–.012rad，参考仅.00027–.00030rad。50Hz全关节均值掩盖了局部高频问题。

12组只读物理对照已完成，baseline与既有native1799状态最大差异0。kd×1.25只有约13%高频改善；减少kp导致失败；直接加大动作滤波也会增加振荡或导致失败。故不直接用这些未适配变体替换默认模型。

## 本阶段方案

先在既有可微DeploymentMDP中增加可选ankle_rate_weight，默认0，不改Policy结构、action_smoothing=.35、PD增益、物理模型、参考或部署输入。原Policy的.35是新提议的占比alpha，公式previous+alpha*(proposed-previous)，不得误作旧动作占比。

新成本：ankle四关节的电机目标变化速度`scales*(a-a_prev)/.02`与参考qd之差，以2rad/s归一化平方均值；再加.5倍实际ankle qd相对参考qd的同样成本。总成本乘ankle_rate_weight和.02，从奖励中扣除。既约束策略指令的高频输出，也约束实际速度偏离，保留参考的主要脚步。它通过现有reward接口同时进入collector和imagined rollout，无需修改RL核心或添加特权策略输入。

mdp保存action_scales为Python tuple，按action.new_tensor创建常量，避免新增CPU张量在MDP.to后未迁移。weight有限且非负；可独立于balance开启。旧默认完全兼容。若后续证据表明此成本不足，可迭代控制器或参考物理可行性；不能将静止、删掉主体动作或视频后处理当成解决。

## 200Hz验收补齐

NativeSimulator增加默认关闭的step observer：reset后和每个5ms mj_step后提供time/qpos/joint_velocity/root_velocity的复制值。不得调用额外mj_forward或改变积分、传感器同步、PD或控制目标；测试 observer与无observer的物理状态逐步一致，回调改副本不影响原状态。Sequence evaluator可将observer传入，新continuity evaluator保存真实7193状态并结合原完整序列门槛。

连续性metrics对每阶段和覆盖首尾的2秒滑窗计算去线性趋势/Hann窗>8Hz谱能量RMS，使用实际采样间隔并检查完整、单调/均匀时钟。中间工程必要门槛：各阶段ankle HF角RMS≤.0025rad，2秒滑窗≤.005rad；root竖直HF位移阶段RMS≤.0005m，滑窗≤.001m。报告全部200Hz加速度/jerk和落脚冲击，不只均值。门槛是必要条件，不能替代全身视频检查或独自证明目标完成。

目标最终验收仍覆盖：完整站立/进入/舞蹈/退出/站稳；无隐藏reset/钉根/轨迹后滤波；主要动作保真；异常倾倒/限位/扭矩/命令超时守卫；高频/冲击和最差窗口；最终策略在原训练引擎与独立MuJoCo/DDS中的全程表现；多次DDS和摩擦、质量、延迟、噪声组合。当前源训练引擎的最终策略验收尚未补齐，不能只验证同一目标物理后改称全部sim2sim无问题。

## 训练与选择

从已复现完整状态 `finetune_projects/g1_continuous_20261029/run_009_reproduce/last.pt` 热启动；新项目 `finetune_projects/g1_continuity_20261029`，首轮1000iterations、64env、50WMwarmup、LR1e-5、每250保存、ankle_rate_weight=.2，沿用balance1/ground.35/action-rate-.3。重点区间覆盖entry和6–14/20–28秒，保留原首帧/entry-exit/uniform采样。

以完整200Hz反抖门槛和原保真/稳定门槛共同筛选，不用“跌倒次数为0”代替连续性。未满足就保持goal active继续迭代；旧模型、原始轨迹和所有失败变体保留，hardware_ready=false。不连接真实机器人。

## 验收修订与第二轮（实测后）

独立review复现Hann端点盲区：完整28秒序列首/尾0.2秒平滑10Hz单踝振荡，峰值.086rad、速度6.14rad/s，旧窗仍通过。metrics版本2新增首尾半窗的奇反射频谱扩展，将真实端点放在Hann中心，同时保持端点一阶差分连续。扩展仅用于频谱，不写回原始状态/录像/物理加速度指标；低频2Hz不同相位回归保证不会由偶反射速度尖点误判。普通重叠2秒窗和各阶段门槛保持，所有门槛仍须通过。

run0011000轮，四候选250/500/750/1000都完成物理全程，但没有通过连续性与保真共同门槛。指令变化速度误差约减半，最差实际踝HF只降约6%，跟踪RMSE随训练增大。拒绝升级，不继续单纯增加同配方轮数。

额外独立物理探针：原最佳权重在踝kd×2时全程完成、danceRMSE .12227（原.12146）、最差HF .010207rad（原.013773，降25.9%）；kd×3失稳。run001/500配kd×1.5跟踪仍差，×2退出失稳。这支持将可接受阻尼变化纳入训练，而不是部署时临时修改。

run002以simulation-only `data/damping_kd200`训练，四踝kd从1.8144456866到3.6288913732，kp/其它kd/参考/Policy结构不变；manifest记录源hash和profile。仍从run009复现的完整最佳权重热启动，非失败run001末尾。1500轮、64env、WMwarmup100、LR1e-5、每250保存，ankle_rate_weight降至.05，沿用balance1/ground.35/action-rate-.3及相同重点区间。默认部署bundle仍008/200；没有真机参数变更。必须验证训练、导出bundle和DDS使用同一profile。

新增原训练MJWarp探针读取相同1799帧sequence.csv并断言参考一致，记录每个pre-reset transition的实际50Hz状态（含终点），遇到done立即停止，不能把自动reset后的状态当成连续轨迹。现默认模型在原训练33.34kg模型完整完成，1799/1799，无物理termination；该探针尚不是200Hz连续性验收，最终新策略仍需重新验证。

## 降速参考适应试验

直接用run002/250在移动阶段speed=.9/.8的参考上测试，两者物理完整，但worstHF .01067/.01636rad、danceRMSE .13145/.14098，均失败，不能把降速当即插即用。源姿态曲线按线性/SLERP采样，standing/settle时长不动；全部重采样q对同曲线的最大差<1.5e-5rad，policy/gains未改，原始动作幅度未缩放。

先用同run002配方重现250轮生成含critic和优化器的完整checkpoint，并逐张量比较Pi/WM与保留250候选，非相同则拒绝继续。然后run004_slow090从该完整checkpoint热启动，在仅移动阶段慢10%的参考上训练800轮，50WMwarmup、LR5e-6、每100保存；其余权重/.05ratecost/kd2一致，重点采样区间相应映射。相同200Hz/保真/稳定门槛；这不是最终采纳决定，Goal仍active。
