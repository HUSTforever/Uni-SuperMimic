# 快速回站与持续站立

本次只验收当前演示的结束段。全段舞蹈的高频抖动与动作保真度仍未全部达标，不能据此宣布整个 sim-to-sim 或真机部署已通过。

## 方法

- 保留原始策略和 PD 参数，保留舞蹈结束帧以前的参考姿态；将原来的平滑退出段缩短为 0.8 秒，随后增加 12 秒静态站姿参考，由同一反馈策略持续控制。未来目标会提前影响舞蹈最后约 0.2 秒。
- 只在退出段施加两端位置、速度、加速度为零的贴地修正，峰值上限 2 cm；转换前检查整个源动作关节限位，转换后再次检查限位、帧数和姿态误差。50 Hz 身份转换直接使用整数帧，避免重插值改变原姿态。
- 正常完成且关节和 IMU 静态检查通过时，DDS 最后一条命令保留最后实际使用的策略目标与原始刚度；异常、无效传感器和旧配置仍使用阻尼保护。保留通信超时看门狗。
- 评价使用完整 200 Hz 原始物理状态；视频使用原始 50 Hz 记录，不平滑轨迹、不重置机器人状态。

## 实测站稳标准

在舞蹈结束后，寻找一直持续到仿真结束的最后一个连续稳定区间，至少确认 0.5 秒。该区间起点距离舞蹈结束不得超过 1.5 秒，而且最后 10 秒必须全部满足以下条件：默认站姿关节 RMSE < 0.10 rad、关节速度 RMS < 0.30 rad/s、根节点速度 < 0.10 m/s、倾斜 < 0.20 rad、根节点高度 ≥ 0.35 m。另检查完整物理运行、退出段高频连续性和正常终止控制模式。

这是有限时长物理验证。关节和 IMU 静态检查本身不能证明接触稳定；批处理仿真完成后会关闭进程，并不提供停止通信后的无限期站立保证。

## 对比与取舍

`FAST_FINISH_NATIVE_RESULTS.json` 保存两个来源、0.8/1.0/1.2 秒退出指令的六组名义质量原生仿真；这些名义质量结果仍有效，原退出段约需 2.2 秒。当前结束预览采用 0.8 秒收势指令加 12 秒主动保持，更正后的名义/扰动回稳约 1.12/1.18/1.20 秒，末 10 秒均稳定；证据入口为 `FAST_FINISH_CORRECTED_NATIVE_RESULTS.json`。

旧的非单位质量缩放实验（质量乘 1.05）没有在缩放后更新 MuJoCo 质量缓存，组合扰动结论因此失真。`FAST_FINISH_STRESS_RESULTS.json` 和 `FAST_FINISH_V2_NATIVE_RESULTS.json` 中的非单位质量行已被 `CORRECTED_MASS_STRESS_RESULTS.json` 替代，旧文件保留作历史；原名义质量和 DDS 结果仍有效。按 `case` 字段区分：`CORRECTED_MASS_STRESS_RESULTS.json` 的 `preview` 两行是当前结束预览，均已通过；`baseline` 两行属于另一套集成基线 bundle（seed 21 未通过、seed 22 通过），不能算作当前 `preview` 的失败。

`FAST_FINISH_DDS_RESULTS.json` 保存第一版快速动作的三次 DDS 复验，回稳 1.135/1.125/1.125 秒，最后 10 秒均通过，终止模式均为 `standing_hold`。实际终止发布者的故障注入仍触发 `LowCmd timeout; damping latched`。

修复身份重采样误差后重新生成的版本为 `data/fast_finish_experimental_0800_v2`。相对来源 `candidates/project006_v3_000200`，主动作关节和根位置误差均为零，根旋转矩阵最大误差约 4.77e-7。其名义质量原生结果仍有效；两组非单位质量组合扰动行已由 `CORRECTED_MASS_STRESS_RESULTS.json` 替代，旧文件保留作历史。最终 DDS、选择和视频证据见 `FAST_FINISH_SELECTION.json`。

此前三次 DDS 回稳时间为 1.125/1.140/1.120 秒，均通过末 10 秒保持和退出段连续性检查，正常终止模式为 `standing_hold`，未触发故障。该版本的真实断联测试仍触发超时阻尼。当时常规测试 334 通过、2 跳过、9 项 slow 未运行；相关代码 Ruff 检查与格式检查通过。旧录像 `artifacts/fast_finish_0800_v2_tail.mp4` 保留。

2026-10-30 追加完整 DDS 回环复测（仅 `lo`、domain 121）：2038 个控制步、8153 个 200 Hz 原始物理样本全部完成；动作结束后 1.135 秒进入持续到片尾的稳定站姿，末 10 秒全部稳定，退出段连续性通过，终止命令为 `standing_hold`，没有物理故障或看门狗触发。完整证据见 `FAST_FINISH_USER_RECHECK.json`；同日常规测试 373 通过、2 跳过、9 项 slow 未运行。新录像 `artifacts/fast_finish_user_recheck_121_tail.mp4` 为原始状态的 50 fps 渲染，960×540、691 帧、13.82 秒；包含最后 1 秒舞蹈、0.8 秒收势和 12 秒主动站立，已完整解码并检查收势与站立画面。

该结果只针对当前结束预览，不是整体部署选型；全段连续性/保真度仍未全部合格，且验证为有限时长仿真，不提供停机后的无限期站立保证。

## 复现

从仓库根目录运行。所有输出目录都必须尚不存在，旧产物保留。只使用仿真回环网卡 `lo`；不连接真机。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unisupermimic.finetune.finish \
  --source-bundle finetune_projects/g1_continuity_20261029/candidates/project006_v3_000200 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --output finetune_projects/g1_continuity_20261029/data/fast_finish_rebuild \
  --exit-seconds 0.8 --hold-seconds 12

OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m unisupermimic.deploy.run_sim2sim \
  --bundle finetune_projects/g1_continuity_20261029/data/fast_finish_experimental_0800_v2 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --transport dds --init sequence --domain 119 --no-video \
  --output finetune_projects/g1_continuity_20261029/evaluation/fast_finish_replay
```

去掉 `--no-video` 可在完整运行后渲染真实仿真视频。独立的回稳评价接口为 `unisupermimic.finetune.finish.summarize_finish`，输入 DDS 的 `simulator_trajectory.npz`，时间为 `(tick - tick[0]) * 0.001`；原生评估使用 `physics_trajectory.npz` 中的 `sample_time`。
