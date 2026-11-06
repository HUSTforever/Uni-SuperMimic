# 200Hz真实高频成本与难片段采样

完整目标和原始验收门槛不变，快速回站要求保留。上一轮为progress：真实PPO两组400轮16候选均完整运行、最佳HF改善9.4%，仍未达标。下一轮基于反复出现在19.5～27秒的高频峰值，对比频段匹配的真实物理成本与难片段采样，不只延长旧奖励训练。

## 方案与边界

采用200Hz实际位置跟踪误差的四阶Butterworth高通成本，截止8Hz；四个踝关节与root-z分别归一化到.005rad/.001m，每个20ms控制步平均其4个真实子步的平方成本。误差为实际pose减线性插值的原参考pose，因此保留原参考的动作变化。两个二阶节用Q=1/(2cos(pi/8))和1/(2cos(3pi/8))；高通系数参照 https://www.w3.org/TR/audio-eq-cookbook/ ，使用NumPy实现，不引入SciPy依赖。

成本仅用于PPO训练回报，不过滤执行动作、状态、视频或修改验收指标。原PD、策略结构、归一化、参考、源文件与旧产物保留。可选成本默认0，默认采样不变。

## 状态和回报契约

HighpassBank(num_envs,channels=5,cutoff_hz=8,sample_hz=200)维护每个环境两节转置DFII状态；reset(row,values,velocity)以当前误差和速度误差初始化，常量/匀速误差不产生伪振荡。observe(row,values)消费单个5ms样本；一行重置不能影响其他行。有限输入/参数必须检查。

SpectralCollector继承现有DeploymentCollector，仅通过NativeSimulator原有step_observer读真实子步。base新增受保护_make_simulator工厂，默认返回完全相同的NativeSimulator。观察器不调用forward/step、不修改真实状态。每次step恰好4个样本（包括5+15ms延迟分支），按当前f与子步比例插值参考。filter在每个自动/手动reset按真实初态重建；完成当前步的成本在auto-reset后仍须保留并计入终止回报。

PPO消费的transition.r与state.reward都减去.02*(ankle_weight*mean_ankle_cost+root_weight*root_cost)。episode_return对未done行减当步成本，done行已保存ep_ret_norm补减当步成本，避免漏算/重复历史。记录两个成本及总惩罚供审计；物理状态与baseline完全一致。

PhysicalPPOConfig增加ankle_hf_weight/root_hf_weight（默认0，有限>=0），focus_start/focus_end（默认0/0禁用，否则整数0<=start<end<=reference末帧）。只在HF权重非零时选择SpectralCollector，其余用原Collector；focus复用现有采样接口且不缩短验收范围。CLI由现有dataclass字段自动生成，配置和checkpoint保存完整参数。

## 对照与验收

TDD：常量/线性误差无伪HF；2/8/12/40Hz正弦增益符合四阶高通解析式；环境隔离和无效输入；实际collector在相同动作/重置/延迟下状态严格等于baseline、样本数4、reward/episode终态成本守恒；新PPO配置路由与focus边界/default不变。常规suite与一次新上下文审查。

先离线重放已保存200Hz轨迹确认成本大小与原频谱指标关系，然后同源检查点/同seed比较两组：focus相同（19～27.96秒），HF权重0与踝.2/root.2，均保留原奖励及实测速度倍率20。源为physical009/175及其完整PPO检查点，原始动作保真仍按固定原anchor判断。各200轮/每25保存，非单调则按完整独立物理评估选型。只对原生全项合格候选继续合并快速回站、源引擎/多次DDS/扰动/10秒保持/原始50fps视频验收。无合格候选则保留global goal active，继续基于证据优化，不连接真机。
