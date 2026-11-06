# G1 部署适配微调实验

本次已实际完成仿真强化学习微调。当前舞蹈的真实 SDK2/DDS 回环跟踪从 **0/3 次完成** 提升到 **3/3 次完整完成**；站立进入和结束回站立仍未通过，**不能据此直接上真机**。

后续已把起停一起纳入训练，并通过完整流程验收。最新候选、录像和说明见 [连续起停实验](../g1_continuous_20261029/README.md)；本目录保留当时仅适配舞蹈段的基线与结果。

## 原来是否已有微调方法

有。原仓库的 `training.init_checkpoint` 会调用 `SuperMimicLearner.initialize_weights()`：载入策略 `pi`、世界模型 `wm`，有 critic 时也载入；重新建立优化器、迭代计数和经验池。之前舞蹈项目 `run_002` 已经使用这个入口。

另一个入口 `algo.resume=true` 是恢复完整 `last.pt`，会恢复优化器和计数，更适合接着原实验继续跑。更换物理条件做新实验时，本轮采用前一种权重热启动。

本轮新增的是 **面向部署差异的实验配方和采样器**。这是一轮使用现有 SuperMimic 世界模型方法的仿真 RL 参数更新，没有把它称作作者官方的额外微调阶段，也没有用真机采集或在线学习。

## 为什么这样微调

原训练模型约33.34kg，部署采用的官方G1模型约35.11kg；关节armature、阻尼、摩擦损失也不同。原训练使用位置伺服，独立部署使用200Hz显式PD和50Hz策略。旧模型在无通信延迟的参考状态初始化下可以跟踪整段，但5ms延迟、零初速度或真实DDS会在约16秒失败。

因此本轮保持参考动作、舞蹈节奏、关节范围和策略接口不变，让训练也接触部署的物理模型/PD及受控扰动。

## 本轮微调方法

1. **热启动，保留旧产物。** 新实验位于本目录 `run_001/`。从舞蹈 `run_002/last.pt` 读取策略、世界模型和critic；已逐项验证其pi/wm与旧选中 `ckpt_001000.pt` 完全相同。丢弃旧优化器、计数器、经验池。旧检查点不修改。
2. **对齐物理采样。** 新的 `DeploymentCollector` 直接使用部署的 `NativeSimulator` 和官方G1资产，显式PD每5ms计算一次、策略每20ms更新。电机力矩和关节目标限幅与部署一致；下一步观测反馈实际限幅目标对应的raw action。策略观测仍是99维、10帧历史，目标730维，输出29维raw action，内部平滑系数保持0.35。
3. **先适配世界模型。** 前150轮只更新世界模型，之后再更新策略和critic。策略观测/目标归一化统计在整轮实验冻结；回归测试确认预热阶段策略参数和归一化buffer均不变。世界模型归一化保留源统计并继续更新。
4. **小学习率继续RL。** 学习率5e-5，1000轮，64个CPU MuJoCo采样环境，GPU学习。每轮4个世界模型batch，batch512；策略batch512，想象rollout32步，经验池600个时间步。初始动作探索噪声降到0.05。策略想象中的动作也投影到部署关节范围，避免优化现实无法执行的目标。
5. **有针对性的扰动和采样。** 每次reset随机摩擦0.7–1.1；25%零初速度；0/5ms动作延迟各半。延迟行在首步之后先保持旧目标5ms，再执行新目标15ms。reset采样25%首帧、25%已知困难区间14–18秒、50%整段均匀采样，并加入小幅姿态/位置/速度扰动。困难区间来自已有失败，属于适配数据，不是未见测试集。
6. **按固定规则选模型。** 保存250/500/750/1000轮，固定测试参考起跑、5ms延迟、零初速度三个场景。先比较完整通过场景数，再比较最差完成比例，最后比较平均关节RMSE。随后按“源0→微调0→源1→微调1→源2→微调2”运行六次真实双进程DDS，控制器、模型、限幅和参考均不改变。

正式训练共164096条物理transition；训练循环324.13秒，含初始化总计327.66秒；峰值CUDA已分配显存约2.42GB。策略学习参数变化RMS为0.00162665，归一化buffer保持逐位相同。

## 结果

| 相同验收条件 | 源策略 | 微调1000轮 |
|---|---:|---:|
| 参考起跑，原生物理 | 1048/1048，RMSE0.1642rad | 1048/1048，RMSE0.1154rad |
| 5ms延迟，原生物理 | 819/1048，失败 | 1048/1048，RMSE0.1134rad |
| 零初速度，原生物理 | 807/1048，失败 | 1048/1048，RMSE0.1156rad |
| 实际DDS完整跟踪 | 0/3次，均817步失稳 | 3/3次，均1048步完整完成 |
| 摩擦0.7、质量×1.05、关节位置噪声σ0.003rad的组合 | 788/1048，失败（此前保留结果） | 1048/1048，跟踪通过 |
| 站立进入 | 未通过 | 未通过，进入策略前即失败 |
| 结束回站立 | 未通过 | 三次DDS均未通过 |

关节误差的完整原生对照下降约29.7%。对实际DDS**共同前818个已记录状态**做公平比较（50Hz，3次均值）：关节RMSE **0.1650→0.1012rad**，加速度RMS **31.35→15.29rad/s²**，jerk RMS **1780.55→875.75rad/s³**。没有把旧策略失败后缺少的尾段填成成功或用于相同长度比较，也没有对录像轨迹做平滑。各轮DDS实际状态采样间隔在本次记录中均为20ms。

所有250/500/750/1000轮候选均通过三个原生场景，平均RMSE依次为0.13716、0.11596、0.12204、0.11483rad，按预先固定排名选1000轮。新导出包明确标注 `candidate_status=unvalidated`、`accepted_for_delivery=false`、`hardware_ready=false`，不会因跟踪改善自动变成真机许可。

## 文件

- `run_001/finetune_config.json`：完整来源hash、训练参数和采样配方。
- `run_001/train_log.csv`、`training_result.json`：实际训练记录。
- `run_001/ckpt_000000.pt`、250/500/750/1000检查点、`last.pt`：保留全部候选。
- `candidates/ckpt_001000/`：当前选中策略与原封不动的参考NPZ。
- `selection.json`：在DDS比较前固定的候选排名与选择。
- `evaluation/source_native/`、`native_000250/`等：固定原生场景。
- `evaluation/dds_source_0/`至`dds_finetuned_2/`：六次交替实测、原始轨迹和日志。
- `evaluation/finetuned_standing/`、`finetuned_perturbation/`：独立补充验收。
- `artifacts/finetuned_dds.mp4`：实际DDS第0次微调策略物理轨迹，50fps；只包含跟踪阶段。
- `RESULTS.json`：汇总和共同片段连续性指标。

大权重、轨迹和媒体留在本地并由gitignore排除；源码、配方、摘要报告保留。

## 复现命令

在 `/home/ytkj/cx/supermimic` 运行，选择未存在的新目录。训练会重新从旧的源模型开始同一轮微调：

```bash
bash finetune_projects/g1_deployment_20261029/train.sh run_002
```

通用训练入口可调整迭代轮数、batch、学习率等，完整默认配方见 `train.sh`。对新检查点导出明确标记的候选包：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unisupermimic.finetune.export_candidate \
  --source-bundle deploy/artifacts/ji_ni_tai_mei/bundle_v3 \
  --checkpoint finetune_projects/g1_deployment_20261029/run_001/ckpt_001000.pt \
  --output finetune_projects/g1_deployment_20261029/candidates/reexport_001
```

固定的三个原生验收场景：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m unisupermimic.finetune.evaluate \
  --bundle finetune_projects/g1_deployment_20261029/candidates/ckpt_001000 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --output finetune_projects/g1_deployment_20261029/evaluation/recheck_native_001
```

实际SDK2/DDS回环；舞蹈跟踪完成后会继续尝试回站立。当前退出阶段失败，因此此命令返回非零退出码，需分别读 `tracking.completed` 和 `exit.completed`：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m unisupermimic.deploy.run_sim2sim \
  --bundle finetune_projects/g1_deployment_20261029/candidates/ckpt_001000 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --transport dds --domain 17 --no-video \
  --output finetune_projects/g1_deployment_20261029/evaluation/recheck_dds_001
```

## 限制与下一步

这轮只有一个训练seed、一个舞蹈，不能推断所有舞蹈或所有扰动都已适配；旧策略仍保留，未验证其它舞种是否遗忘。5ms固定子步延迟是本机DDS延迟的近似，世界模型没有额外显式编码执行器目标记忆，不能覆盖任意网络抖动。组合扰动只做一次补充仿真，不代表普遍鲁棒性。

站立阶段当前仍使用已有的简单PD目标渐变，尚未改为官方独立站立/动作切换流程。它发生在策略开始前，单独增加舞蹈训练轮数不会直接修复该阶段。下一步应对齐站立状态机、首尾平衡过渡，再验收完整流程；仍未做实际机器人的标定和保护测试。

代码验收：240项默认测试通过、2项跳过、8项slow未选；新增部署实际DDS slow测试另外2项通过。独立代码审查未发现Critical/Important问题。所有实际DDS仅本机lo和非零domain，未连接真机。
