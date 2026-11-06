# G1 连续动作：当前贴地优化版

**当前选中 `run_008/ckpt_000200.pt` / `candidates/ground008_000200/`。** 恢复后做了4000轮对照训练、16个检查点比较，最终选择中间权重的早停点；另用200轮精确复现，保存完整热启动状态。旧模型/录像/结果全部保留。

新模型在实际SDK2/DDS本机回环连续3/3次完成完整站立→舞蹈→退出→站稳，200Hz物理、50Hz控制，1798步/1799个真实状态；5ms延迟与摩擦.7、质量×1.05、位置噪声σ.003rad组合场景也通过。没有连接真机，`hardware_ready=false`。

与上一默认run003/2000比较：

| 指标与口径 | 上一版 | 当前版 | 变化 |
|---|---:|---:|---:|
| native承载接触点速度RMS（法向力加权，50Hz边界） | .2243m/s | .1988m/s | 降11.3% |
| DDS参考支撑点水平运动RMS（200Hz、三次均值） | .0510m/s | .0431m/s | 降15.6% |
| DDS全部近地点运动RMS | .4625m/s | .4479m/s | 降3.1% |
| DDS机身重力方向误差RMS | 1.379° | 1.374° | 基本保持 |
| DDS机身角速度误差RMS | .1453rad/s | .1459rad/s | 增.4% |
| DDS机身角加速度RMS | 9.469rad/s² | 9.188rad/s² | 降3.0% |
| DDS舞蹈关节跟踪RMSE | .1183rad | .1215rad | 增2.7% |

本轮主要改善脚部滑动，整体动作连续性只有小幅改善：同一核心949状态按实际时间差分，加速度13.40→13.09rad/s²，jerk748.85→744.52rad/s³，分别降2.3%/.6%，不夸大为大幅消抖。参考动作、节奏、主体路径逐字节未变；没有对物理轨迹做后期平滑。

- [当前完整录像](artifacts/ground008_200_dds.mp4)、[完整前后对比](artifacts/ground008_before_after.mp4)、[6–14秒重点对比](artifacts/ground008_6_14_compare.mp4)。
- [本轮微调方法](GROUND_METHOD.md)、[最终结果和hash](GROUND_RESULTS.json)、[候选明细](ground008_candidates.json)。所有未选中的试验保留。
- 后续热启动使用 **`run_009_reproduce/last.pt`**：pi/wm与选中200轮检查点逐张量完全相同，同时保留该时刻critic/优化器状态；复现验证在`ground_reproduction.json`。`run_008/last.pt`是未选中的500轮，不应误当成选中模型。训练入口仍按新run重建经验池/优化器。
- `evaluation/dds_ground008_200_initial/`及`dds_ground008_200_repeat1/2/`为三次原始DDS记录，均无watchdog或物理失败；`ground008_200_delay5/perturbation/`为额外条件检查。
- 261项常规测试、3项实际DDS集成通过；手动CPU/CUDA非零奖励与梯度一致，源权重/参考SHA及策略归一化未变。

脚部指标分别是原生仿真接触解诊断与DDS运动学代理，不能相互冒充，也不能据此保证所有工况或真实机器人的稳定性。模型包仍保留candidate标记，验收由GROUND_RESULTS绑定具体文件hash。

当前DDS命令可用下方历史示例，将bundle换成`candidates/ground008_000200`并使用新输出目录，保留`--init sequence`。继续训练的参数必须包括`--balance-weight 1 --ground-slip-weight .35 --action-rate-weight -.3 --lr 1e-5`，完整配方见run008配置和GROUND_METHOD。

---

# 上轮历史：动态平衡追加训练

**上轮选中 `run_003/ckpt_002000.pt` / `candidates/balance_002000/`，现作为回退保留。** 用户反馈原视频仍有站不稳后，追加两轮共3000轮训练，比较6个检查点；选中2000轮模型，后续1000轮未整体优于它。完整站立→舞蹈→退出→站稳在真实SDK2/DDS本地回环3/3次通过，5ms延迟/组合扰动也通过。未连接真机，`hardware_ready=false`。

新旧各3次DDS、相同舞蹈段5–27.96秒、200Hz真实轨迹均值：

| 指标 | 旧平滑版 | 本轮选中版 | 变化 |
|---|---:|---:|---:|
| 机身重力方向偏差RMS | 2.62° | 1.38° | 降47.3% |
| 机身角速度偏差RMS | .240rad/s | .145rad/s | 降39.5% |
| 机身角加速度RMS | 11.16rad/s² | 9.47rad/s² | 降15.1% |
| 参考支撑点水平运动RMS | .0724m/s | .0510m/s | 降29.5% |
| 全部近地球心水平运动RMS | .4567m/s | .4625m/s | 增1.3% |
| 舞蹈关节跟踪RMSE | .1095rad | .1183rad | 增8.0% |

本轮以少量跟踪精度换取更少晃动；全部近地脚部运动没有一起改善，不能称为完全消除了滑脚。支撑点指标是运动学代理，不代表接触力或通用稳定性证明。原参考的主要舞蹈路径、节奏、倾身和转身保留，reference逐字节未变。

同一核心949个50Hz状态，按实际时间间隔计算：关节加速度RMS14.14→13.40（降5.2%），jerk823.10→748.85（降9.0%）。一次新DDS核心含两对25/15ms间隔，其余为20ms，另存名义50Hz口径以便对照；没有对物理轨迹做后期平滑。

- [完整新录像](artifacts/balance_run3_2000_dds.mp4)、[完整前后对比](artifacts/balance_before_after.mp4)、[20–30秒重点片段对比](artifacts/balance_hard_comparison.mp4)。
- [训练方法与两版奖励出处](BALANCE_METHOD.md)、[筛选规则](BALANCE_SELECTION_RULE.md)、[验收与限制](BALANCE_VERIFICATION.md)、[全部数值与hash](BALANCE_RESULTS.json)。
- `evaluation/dds_balance_run3_2000/`及`dds_balance_run3_2000_repeat1/2/`：三次完整原始DDS轨迹、终点状态、报告及200Hz指标。所有1798步完成，无watchdog/物理失败。
- `run_003`首版脚部判定仅用水平速度，完整源码快照保留；审查后改为三维速度排除快速抬落脚，`run_004`追加1000轮。本轮选中旧判定训练出的2000轮权重，因为实际比较更好；最终指标统一使用修正后三维判定，未混淆两版出处。
- 254项常规测试、3项实际DDS集成通过。代码默认不启用新奖励，已有部署接口和限幅保持一致。

重跑当前模型：使用下方历史DDS命令，将bundle改为 `finetune_projects/g1_continuous_20261029/candidates/balance_002000` 并使用新的output目录，保留`--init sequence`。

---

# G1 连续起停与舞蹈：完整 sim2sim

**以下为前两轮历史记录：原平滑回退模型为 `run_002/ckpt_000600.pt` / `candidates/smooth_000600/`。**

第一轮稳定版（run_001第1250轮）连续5/5次完整通过；进一步减少动作变化的平滑版连续3/3次完整通过。所有试验控制1798步、35.96秒，记录1799个真实物理状态，包含终点。录像为1799帧@50fps、35.98秒（包含终点的显示帧）。验收范围是本项目的官方G1模型和下述条件，未连接真机，`hardware_ready=false`。

## 本轮解决的问题

旧流程用固定PD目标插值处理启动和结束，在静态站姿也会约1.4秒后失去平衡；换成官方FixStand参数仍复现。它不是闭环平衡策略，之前单独训练舞蹈不能覆盖这些阶段。

本轮把首尾一起加入参考动作，用同一个策略连续控制全部阶段。原视频首帧已接近1m/s，故只对舞蹈最初/最后各约1秒做C2渐进变速，速度由0升至原节奏，再由原节奏降为0；保持原几何动作路径，中间动作和节奏不变。由于50Hz浮点重采样，中间参考的最大数值误差约1.6e-5rad，生成器按2e-5rad容差记录。首尾姿态使用五次多项式与SO(3)切线插值，起止速度/加速度为零。

| 阶段 | 时间 | 控制内容 |
|---|---:|---|
| 初始站立 | 0–2s | 静止站姿开始，实际初速度为零，闭环平衡 |
| 进入 | 2–5s | 渐入舞蹈首姿态 |
| 舞蹈 | 5–27.96s | 原舞蹈主体，头尾渐进变速 |
| 退出 | 27.96–30.96s | 渐变回站姿 |
| 稳定保持 | 30.96–35.96s | 闭环站稳5秒 |

实际仿真中只在开始初始化一次，后续所有姿态来自自由基座物理仿真和策略/电机PD控制。完整录像包含起停，不仅是舞蹈段。

## 迭代方法与结果

1. **连续流程训练**：从上一轮部署适配 `g1_deployment_20261029/run_001/last.pt` 热启动，1500轮，前150轮仅适配世界模型，学习率5e-5，64个环境、batch512，保留原策略结构/归一化/0.35动作平滑。25%采样首帧，25%采样进入或退出，50%全段采样。实际训练约459秒，保留所有检查点。
2. **固定标准选稳定版**：500/750/1000/1250/1500轮均通过完整原生仿真及末尾保持，按完成情况、最差阶段关节RMSE选择1250轮；实际DDS5/5次完整通过。
3. **再降低抖动**：从第一轮last.pt重新热启动600轮，学习率2.5e-5，前50轮世界模型预热，将动作变化惩罚从-0.1加大到-0.3，200轮保存一次。实际训练166秒。三个候选均通过完整原生流程，第600轮在跟踪精度与核心加速度/jerk指标上最好，实际DDS3/3次完整通过。
4. **独立验收**：实际DDS使用两个进程、官方G1模型、官方SDK2 HG消息及CRC、本机lo与非零domain，200Hz电机伺服、50Hz策略。报告逐阶段记录完成和误差，并检查最后一秒的实际速度、倾角和站姿误差。所有成功试验包含实际终点状态，避免仅检查末帧之前的数据。

末尾保持门限：关节速度RMS<0.3rad/s、机身最大线速度<0.15m/s、最大倾角<0.3rad、站姿关节RMSE<0.2rad。每个阶段关节RMSE也必须<0.2rad。

最终平滑策略的三次DDS舞蹈段关节RMSE约0.109rad。末尾关节速度RMS约0.0021rad/s、机身最大速度约0.0013–0.0014m/s、倾角约0.012rad（约0.7°），均在门限内。5ms延迟场景与摩擦0.7、质量×1.05、关节位置噪声σ0.003rad的组合场景也通过完整流程及站稳检查。

实际观测为名义50Hz，最终三次中的两次各出现一对25ms/15ms间隔（偶发±5ms抖动），完整流程仍通过。所有样本时间单调且包含35.96秒的真实终点，原始时间戳保留；不是将实际调度假定为完全无抖动。

对比同一舞蹈核心949个状态（原动作50–998帧，新序列350–1298帧），最终DDS三次均值：加速度RMS约14.14rad/s²、jerk RMS约823.10rad/s³；上一版舞蹈模型为15.42、898.02，分别再下降约8.3%。没有通过静止段稀释比较，也没有对视频轨迹做后期平滑。

## 当前产物

- `candidates/smooth_000600/`：当前选中模型、参考、接口metadata；`run_002/ckpt_000600.pt`是原检查点。
- `candidates/selected/`：首轮第1250轮稳定回退版，保留5次完整DDS证据。
- `data/sequence_v2/`：当前参考与生成场景hash；其reference.npz与训练用v1逐字节相同。v2补充单独的生成场景/资产来源字段。
- `evaluation/dds_smooth_0/`至`dds_smooth_2/`：最终3次真实DDS记录，正常退出码0。
- `evaluation/dds_sequence_0/`至`dds_sequence_4/`：第一轮5次成功记录。
- `evaluation/final_smooth_delay5/`、`final_smooth_perturbation/`：完整延迟/组合扰动验收。
- `evaluation/baseline_verified/`：未训练首尾前的失败；早期`native_*`为旧验收草稿，最终采信含终点的`verified_native_*`。
- `RESULTS.json`、`selection.json`、`final_selection.json`：结果汇总、两轮模型选择与连续性口径。
- `artifacts/continuous_g1.mp4`：50fps全流程录像；`start.png`、`preview.png`、`end.png`为起点/舞蹈/终点抽帧。

模型包保留生成时的candidate标记；验收通过状态以`RESULTS.json`绑定的检查点、模型和参考hash以及实际报告为准。未把仿真通过自动标成真机就绪。

## 重跑验收

从 `/home/ytkj/cx/supermimic` 执行，每次选择不存在的输出目录：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m unisupermimic.deploy.run_sim2sim \
  --bundle finetune_projects/g1_continuous_20261029/candidates/smooth_000600 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --transport dds --init sequence --domain 17 --no-video \
  --output finetune_projects/g1_continuous_20261029/evaluation/recheck_001
```

必须使用`--init sequence`，让策略控制完整参考的首尾，避免回到旧的固定PD启动/退出流程。原有`reference`/`standing`模式仍保留作旧基线诊断。

## 训练复现

第一轮从旧的舞蹈适配模型重新建立独立实验：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unisupermimic.finetune.train \
  --bundle finetune_projects/g1_continuous_20261029/data/sequence_v2 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --checkpoint finetune_projects/g1_deployment_20261029/run_001/last.pt \
  --output finetune_projects/g1_continuous_20261029/run_new_sequence \
  --iterations 1500 --num-envs 64 --wm-warmup 150 --batch 512 --wm-batches 4 --lr 0.00005
```

第二轮平滑训练：同一入口改为从本项目`run_001/last.pt`初始化，新输出目录，`--iterations 600 --wm-warmup 50 --lr 0.000025 --checkpoint-every 200 --action-rate-weight -0.3`。完整来源hash和参数在两轮`finetune_config.json`中。权重、经验池和优化器都保留在新目录，未覆盖旧产物。

代码验收：243项常规测试通过、2项跳过；3项真实DDS slow集成测试单独通过。独立审查发现的终点采样遗漏已用回归测试修复。新的DDS集成测试隔离到独立CLI进程，避免官方SDK全局domain单例污染其它测试。

当前结论覆盖上述模型、动作、初始化和扰动范围；尚未进行实际机器人标定或真机试验，也未证明其它舞蹈或任意网络延迟的稳定性。
