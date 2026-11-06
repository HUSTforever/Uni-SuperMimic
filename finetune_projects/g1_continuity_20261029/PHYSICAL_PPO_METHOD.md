# 真实物理 PPO 微调

这是在现有 SuperMimic 策略上增加的实验性微调路径。策略梯度使用实际部署 MuJoCo 的采样回报，算法采用 [PPO clipped surrogate](https://arxiv.org/abs/1707.06347)。原世界模型权重保留在检查点中，训练不调用它计算策略梯度，也不更新它。

## 输入和训练

源包 `candidates/run002_000250`，完整检查点 `run_003_reproduce/last.pt`；舞蹈参考、控制器、PD、策略网络结构与归一化保持原样。启动时逐张量核对源 TorchScript 的全部权重与归一化缓存，同时验证图中的常量行为。策略仍然是原先的确定性均值策略；训练在该均值上添加固定标准差 0.03 的高斯探索，logprob 记录未投影动作，执行时才做限位投影。部署导出使用确定性均值。

复用真实 DeploymentCollector。默认 32 个环境，每轮各采样 32 个 20 ms 控制转移；每个控制转移包含 4 个 5 ms 物理子步。200 轮合计 204,800 个控制转移、819,200 个物理子步。环境随机化初态、摩擦和 0/5 ms 延迟。回报使用实际状态，自动重置前的 s1 用于 bootstrap；真实失败或参考结束不 bootstrap，时间上限 bootstrap 但 GAE 不跨越重置。

固定 gamma=.99、lambda=.95、clip=.1、策略 LR1e-5、critic LR1e-4、梯度范数上限1、近似KL停止阈值.015。每轮4epochs/batch256，先10轮只更新critic。锚定初始策略以限制漂移；每25轮保存候选，最终保存完整PPO优化器。PPO检查点携带原WM与原导出args，独立记录PPO配置，避免混淆两种训练路径。

## 验证

初始导出的全段物理结果复现原来源：最差踝部HF .0089697125 rad、原始舞蹈关节RMSE .1269085135 rad。相同真实状态下1798个控制步的接口核对：目标角误差最大2.09e-7 rad，原始动作误差9.54e-7，未来goal完全相同。独立运行的轨迹在26.8秒以后出现小量数值放大，最大关节差.009216 rad、根位置差1.605 mm；两次均完整运行，不声称逐状态完全一致。

首轮 `run_008_physical_ppo` 使用原奖励。200轮、8个候选均完整运行，但没有全项通过。最佳25轮最差HF .008456 rad（约改善5.7%）、舞蹈RMSE .125732、原始幅度与相关性均通过；200轮退化为HF .010942、RMSE .131968，未采纳。结果见 `PHYSICAL_PPO_RESULTS.json`。这证明了训练链路可运行，也证明不能依据训练奖励或最终检查点选型。

检查点逐张量核对证明初始pi与来源严格相同、最终WM与策略归一化严格不变、actor实际更新，来源文件hash不变，见 `PHYSICAL_PPO_TRAINING_VERIFICATION.json`。新上下文审查发现的固定探针身份检查缺口，已通过SVD零空间反例复现并修复。

## 实际速度项对照

256个真实随机舞蹈采样步的奖励审计见 `PHYSICAL_PPO_REWARD_AUDIT.json`。踝成本中指令变化误差均值2.2895，实测速度误差均值.2322，但后者在原公式中仅乘.5；原踝权重.05，因此实际速度项只占踝惩罚约4.83%，占tracking奖励约0.128%。

第二轮 `run_009_physical_measured` 只将实际速度部分乘20。源检查点、参考、PD和其余超参数与第一轮完全相同，作为受控对照。默认倍率仍为1；旧WM训练路径仍使用原始成本。结果记录于 `PHYSICAL_PPO_MEASURED_RESULTS.json`，同样须通过所有原始门槛才能进一步验证DDS、扰动和快速回站。

第二轮200次更新和8个候选的独立物理评估已完成。最佳175轮最差HF .008126 rad，相对初始下降9.4%，原始舞蹈RMSE .126460、主要关节幅度/相关性通过；仍超过窗口.005 rad及部分阶段门槛，不采纳。两组共16个候选均完整物理运行，零个全项合格。初始pi/critic/WM在两组间逐张量一致，配置仅倍率1→20改变，见 `PHYSICAL_PPO_ABLATION_VERIFICATION.json`。第二组最终WM和归一化仍严格不变，来源hash不变，汇总见 `PHYSICAL_PPO_SUMMARY.json`。

离线200Hz轨迹分析 `PHYSICAL_PPO_SPECTRAL_AUDIT.json` 显示最差窗口反复位于19.5～27秒。16个候选中，全频实测速度误差与最差HF的相关系数约.636，按现有频段计算的窗口平均HF成本相关约.870；这只提供后续频段目标/难片段采样的对照方向，不能证明唯一根因，也没有验证新的训练奖励。使用现有Hann指标完成该分析，没有安装SciPy或更改验收方法。

当前代码常规测试346通过、2跳过、9项slow未重复；新增GAE/投影概率/重置/权重身份/冻结/导出/倍率测试均经过RED→GREEN。核心PPO已独立审查，重要问题已修复；倍率扩展由主代理检查真实diff并复跑新旧成本测试。整个连续性目标继续active，未对尚不合格的新策略执行部署采纳或真机操作。

## 复现

仓库根目录，输出必须尚不存在：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m unisupermimic.finetune.physical_train \
  --bundle finetune_projects/g1_continuity_20261029/candidates/run002_000250 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --checkpoint finetune_projects/g1_continuity_20261029/run_003_reproduce/last.pt \
  --output finetune_projects/g1_continuity_20261029/run_physical_replay \
  --measured-velocity-multiplier 20 --device cuda
```

原权重、默认策略、原视频和快速回站演示保留。此实验仍须满足完整200Hz连续性、原始动作保真度、多次DDS、扰动、快速回站与真实视频检查，不能把完成训练或不跌倒当作全流程无颠簸的证据。
