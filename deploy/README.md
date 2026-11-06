# G1 29-DoF 部署程序级 sim2sim

已建立独立的策略运行程序和真实 SDK2/DDS 本机回环。**最新连续序列已通过包含起停和站稳的sim2sim验收；本文下方保留早期基线，尚未进行真机测试。**

最新模型、约36秒完整录像和重跑命令见 [连续序列验收](../finetune_projects/g1_continuous_20261029/README.md)：首轮5/5次、最终平滑版3/3次实际DDS完整通过，起停阶段由闭环策略连续控制，使用`--init sequence`。

后续已完成一次部署适配微调：新策略在三次实际DDS中均完成舞蹈跟踪，但站立进入/结束回站立仍未通过。微调方法、新候选模型与前后对照见 [微调实验](../finetune_projects/g1_deployment_20261029/README.md)；本文下方保留微调前的原始基线结果。

这次使用原来的 `run_002/ckpt_001000.pt`，没有覆盖或重新训练旧模型。独立运行程序不调用训练环境、MDP、训练奖励或真值根位置来生成动作。它读取电机位置/速度、pelvis IMU 和预先生成的参考动作，构造 99 维观测、10 帧历史、730 维目标，运行 TorchScript 后产生 29 个关节目标。

## 实测结果（2026-10-29）

物理伺服/状态发布为 200 Hz，策略为 50 Hz。官方模型来自 `unitreerobotics/unitree_mujoco`，commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`。

| 场景 | 整段跟踪 | 启动/退出 | 判断 |
|---|---|---|---|
| 官方 G1 模型，直接 MuJoCo，参考姿态和速度初始化 | 1048/1048 步，RMSE 0.1642 rad | 回站立失败 | 仅参考初始化的跟踪通过 |
| 同一模型，独立双进程 SDK2/DDS 回环 | 本次 832/1048 步，约 16.6 秒倾斜过大 | 未进入退出段 | 部署链路运行，但策略未通过 |
| 参考姿态、零初速度 | 807/1048 步后倾斜过大 | — | 对初始速度敏感 |
| 站立保持再渐变到首帧 | 0 步，进入舞蹈前跌落 | 启动失败 | 首帧 reset 不能代替启动验证 |
| 摩擦 0.7、质量 ×1.05、关节位置噪声 σ=0.003 rad 的组合 | 788/1048 步 | — | 此组合扰动未通过；未单独归因 |
| 发布进程突然退出 | 接收 3 条有效命令后，100ms 看门狗触发 | 锁定阻尼，力矩受限 | 故障保护测试通过 |

DDS 是实际异步程序，失败步数可能随调度变化；前一次试验为817步，不能把单次无延迟原生仿真通过当作鲁棒性证明。原始结果与路径见 `RESULTS.json`。当前所有舞蹈场景的 `all_attempted_phases_passed` 均为 false；未通过会返回非零退出码。

训练 XML 的总质量约33.34kg，官方 XML 约35.11kg，关节 armature、阻尼、摩擦损失也有差异，详见 `model_comparison.json`。模型差异、异步控制时序、初始状态与目标限幅均需要纳入后续微调，当前数据不能单独确定哪个因素导致失稳。

## 文件与边界

- `artifacts/ji_ni_tai_mei/bundle_v3/`：当前推荐导出包。模型和参考 NPZ 与实际测试用的 bundle_v2 **字节哈希相同**；v3仅加强导出时的图执行/历史索引校验。
- `artifacts/ji_ni_tai_mei/final_native_reference/`：独立原生闭环、完整跟踪视频、退出失败报告。
- `artifacts/ji_ni_tai_mei/final_dds_reference/`：真实双进程通信、实际跟踪视频、状态/命令数量、物理失败与阻尼记录。
- `artifacts/ji_ni_tai_mei/final_standing_entry/`、`final_zero_velocity/`、`final_perturbation/`：各自独立的失败证据。站立进入失败的 tracking 轨迹为零行。
- `artifacts/ji_ni_tai_mei/final_watchdog/`：真实独立发布进程执行 `os._exit(0)` 后的超时保护验收。
- `deps/unitree_mujoco/`：本地官方模型仓库，作为忽略的大依赖保留。
- 旧 `bundle/`、`bundle_v2/` 与早期试验结果保留供审计，默认使用 v3。

本项目 DDS 服务是**自建无界面 HG 适配器，加载官方 G1 资产并使用官方 SDK2 序列化/CRC**，不是未经修改的官方仿真器程序。物理位置只用于初始化、离线验收和录像；策略在线输入不包含根位置。

程序只接受 `lo` 和非零 domain，额外把 DDS 数据与发现约束到127.0.0.1。没有真机启用参数、MotionSwitcher 调用、网卡配置或机载服务操作。此处 IMU 契约是 pelvis wxyz 姿态和机体系 gyro；会检查模型主 IMU 挂载位置，不能把 torso IMU 直接当作 pelvis。

## 本机依赖

从 `/home/ytkj/cx/supermimic` 运行以下命令，使用现有 `.venv`。训练与渲染依赖已安装；额外安装的是 cyclonedds 0.10.2 与本机官方 unitree_sdk2py 1.0.1（commit `814556d15970dd2ecf1c9984e845ca02ab07e206`）。SDK、CycloneDDS 仍复用相邻 Unitree 目录，没有搬迁上游工程。

```bash
CYCLONEDDS_HOME=/home/ytkj/cx/unitree/控制/unitree_sim_isaaclab/cyclonedds/install \
  uv pip install --python .venv/bin/python 'cyclonedds==0.10.2'
uv pip install --python .venv/bin/python --no-deps \
  --editable /home/ytkj/cx/unitree/控制/unitree_sdk2_python
```

若换机器，需要先准备相同版本的 SDK/CycloneDDS 和官方 G1 模型资产，不能仅复制 `.ts` 文件。运行时包内参考与控制参数有哈希校验，但外部物理模型仍由 `--scene` 显式选定。

## 复现

每次选择一个**不存在**的输出目录，禁止覆盖。原生跟踪与 DDS 跟踪使用同一 bundle 和 PolicyController：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python -m unisupermimic.deploy.run_sim2sim \
  --bundle deploy/artifacts/ji_ni_tai_mei/bundle_v3 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --transport dds --domain 17 --init reference \
  --output deploy/artifacts/ji_ni_tai_mei/recheck_dds_001
```

改成 `--transport native` 可做直接物理诊断；`--init reference_zero_velocity` 检查零初速度，`--init standing` 检查站立进入。省略 `--no-video` 会按记录的真实物理姿态生成50fps录像，不进行轨迹后处理。报告把跟踪完成、站立进入和退出分别记录；参考状态初始化不会被标成 `standing_entry_verified`。

故障保护测试会启动本机仿真和另一个发布进程，故意让发布进程突然退出：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unisupermimic.deploy.run_sim2sim \
  --bundle deploy/artifacts/ji_ni_tai_mei/bundle_v3 \
  --scene deploy/deps/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --watchdog-probe --domain 19 \
  --output deploy/artifacts/ji_ni_tai_mei/recheck_watchdog_001
```

重新打包需要已验收的检查点、原TorchScript/JSON、CSV/JSON和验收报告。它会校验权重以及非零历史/previous action 下的执行等价性，检查参考元数据与已验收副本、完整帧数，再生成离线参考：

```bash
.venv/bin/python -m unisupermimic.deploy.bundle \
  --policy dance_projects/BV1ct4y1n7t9/artifacts/final_static/pipeline/policy.ts \
  --motion dance_projects/BV1ct4y1n7t9/motion_static/ji_ni_tai_mei_static.csv \
  --checkpoint dance_projects/BV1ct4y1n7t9/run_002/ckpt_001000.pt \
  --acceptance-report dance_projects/BV1ct4y1n7t9/RESULTS.json \
  --scene src/unisupermimic/assets/robots/g1/scene_supermimic.xml \
  --output deploy/artifacts/ji_ni_tai_mei/repacked_001
```

## 真机前剩余工作

首先在与部署一致的模型/低层PD上，加入站立进入、舞蹈结束的平衡过渡，并针对延迟、初始速度、质量/摩擦和传感噪声微调，再反复通过完整 DDS 闭环。随后才是具体机器人型号、固件、IMU/电机标定与实际保护系统的核对，以及有保护的分阶段真机测试。本次没有执行这些真机步骤。
