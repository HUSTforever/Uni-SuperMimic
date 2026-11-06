# 视频 → G1 连贯动作 → 真实策略 demo

本目录提供统一的本地视频/GMR NPZ → CSV → 可选训练 → 物理评估 → TorchScript → 闭环校验入口。
示例片段来源：B站 `BV1FjhE6PEGb` 的12.0–17.1秒「浪子踢球」全身示范。

上游 GVHMR、GMR、人体模型和两个虚拟环境仍在相邻的 `../mocap`，可用 `--mocap-root` 指定；不会自动下载 B 站视频。输入需要是单人全身的连续片段；固定机位使用默认 static，移动机位可选 --camera simplevo。

新版保留主要踢腿动作，把参考片段约 5.07 秒放慢到 6.33 秒，并平滑根姿态、关节轨迹和支撑高度。策略自身以 50 Hz 输出平滑后的 action；训练、评估和导出使用同样的控制逻辑。视频来自真实物理 rollout，不对录制后的关节轨迹做滤波。

## 已修复和增加

- 正确转换 NPZ/MuJoCo `wxyz` 与 CSV `xyzw`，朝向旋转使用世界坐标。
- 贴地依据足部碰撞球最低点。默认 `--contact-mode support` 明确假设至少一只脚接地；跳跃动作可选 `--contact-mode preserve-flight`，在人形和机器人两阶段保留腾空，同时需要可靠的世界高度及接触验证。
- NPZ 源 fps 参与重采样；CSV 旁的 JSON 记录帧率、时长、处理参数和 SHA256，加载器读取每段 fps 并拒绝内容与元数据不一致。
- 评估从 run 配置恢复自定义 motion；显式换参考动作不修改源运行，并保留其他结构检查。
- 新任务通过 `training.init_checkpoint` 仅初始化权重，重新记录配置和优化器；不覆盖旧 run。
- 导出输出语义为 `raw_action`，附带 `.ts.json` 接口元数据；闭环数值不一致、NaN/Inf 或形状错误均非零退出。
- pipeline 原子写入 manifest，记录输入哈希、命令、阶段结果和失败；不覆盖已有 manifest/run。

## 使用

以下命令均在项目根目录执行，使用已安装的 `.venv`。无新增依赖。

```bash
# 从已重定向的 NPZ 开始，创建新的微调运行；2000 是新增轮数。
env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
.venv/bin/python demo/pipeline.py \
  --npz ../mocap/outputs/kick_rave/g1_qpos_grounded.npz \
  --checkpoint runs/demo_kick_001/ckpt_010000.pt \
  --train-iterations 2000 --name kick_rave_smooth \
  --output-dir demo/artifacts/my_smooth_run

# 复用已训练的新策略，重新转换、物理评估、渲染、导出及闭环验证。
# 每次选择新的输出目录；不会覆盖旧产物。
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl OMP_NUM_THREADS=2 \
.venv/bin/python demo/pipeline.py \
  --npz ../mocap/outputs/kick_rave/g1_qpos_grounded.npz \
  --checkpoint runs/demo_kick_smooth_001/ckpt_001000.pt \
  --device cpu --backend mujoco --name kick_rave_smooth \
  --output-dir demo/artifacts/my_smooth_export

# 从本地视频开始。pipeline 传递人体重新贴地参数，并按视频文件名定位 GVHMR 结果。
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
.venv/bin/python demo/pipeline.py \
  --video ../mocap/inputs/kick_rave.mp4 --mocap-root ../mocap \
  --checkpoint runs/demo_kick_001/ckpt_010000.pt --train-iterations 2000 \
  --output-dir demo/artifacts/my_video_run
```

不训练时，`--checkpoint` 是必需的。改变预处理参数后，既有策略可能不适用；评估未完成整段时 pipeline 会失败并保留诊断，不会继续把该模型标记为成功导出。

默认预处理：`--smooth-seconds 0.07 --slowdown 1.25 --face-yaw-deg 315`；新策略 `--action-smoothing 0.35`。这是每个 20 ms 控制步的系数，属于策略的一部分，旧检查点默认 1.0，保持原行为。

## 分阶段操作与诊断

```bash
.venv/bin/python demo/convert_gmr_to_csv.py \
  ../mocap/outputs/kick_rave/g1_qpos_grounded.npz demo/motion/kick_rave_smooth.csv \
  --ground-clamp --smooth-seconds 0.07 --slowdown 1.25 --face-yaw-deg 315

CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl OMP_NUM_THREADS=2 \
.venv/bin/python demo/evaluate_motion.py \
  --ckpt runs/demo_kick_smooth_001/ckpt_001000.pt --backend mujoco --device cpu \
  --output demo/artifacts/new_evaluation.json --video demo/artifacts/new_comparison.mp4

# 只显示实心策略机器人，避免参考鬼影影响观察。
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl OMP_NUM_THREADS=2 \
.venv/bin/python demo/render_policy_video.py \
  --ckpt runs/demo_kick_smooth_001/ckpt_001000.pt --clip kick_rave_smooth \
  --seconds 8 --fps 50 --device cpu --hide-reference \
  --output demo/artifacts/new_policy_only.mp4
```

评估 JSON 包含后端、检查点/动作哈希、物理完成标记、回报、关节跟踪 RMSE、速度/加速度/jerk 的 RMS 与峰值；同名 NPZ 保存真实参考和策略轨迹。不能把相同数值输出的两个短暂失败 rollout 视为动作质量通过；pipeline 单独检查完整片段完成情况，并要求 float32 导出策略完成全片段闭环验证。

## 数据与部署接口

- 原版 `motion/kick_rave.csv`：153 行 × 36 列，30 fps；加载到控制频率后为 254 帧，原文件不修改。
- 新版 `motion/kick_rave_smooth.csv`：191 行 × 36 列，30 fps；加载后为 317 帧。其 JSON 是帧率和内容身份的组成部分。
- CSV 列：根位置 xyz 3 列、四元数 xyzw 4 列、自然关节顺序的 29 列弧度角。
- 模型输入：`obs_hist[B,hist,99]`、`goal[B,73*goal_frames]`，最新观测位于历史末尾。
- 模型输出：`raw_action[B,29]`，目标为 `default_q + context_offset + action_scale * raw_action`；需要观测和未来参考动作相位同步。
- `.ts.json` 描述关节顺序、默认姿态、动作缩放、控制周期、输入维度及动作平滑系数。模型不包含真机状态采集、参考相位推进或 Unitree SDK 控制桥。

原来的 `runs/demo_kick_001` 和 `artifacts/01_*` 至 `07_*` 均保留。新的验证与交付结果保存在 `artifacts/smooth_final/`，其 manifest/evaluation 是最终实测依据。

## 本次最终实测（同一 CPU/MuJoCo，50 Hz）

| 指标 | 原版10000轮 | 新版微调1000轮 |
|---|---:|---:|
| 完整片段，无失败 | 是 | 是 |
| 关节加速度 RMS (rad/s²) | 45.425 | 11.282（降低75.2%） |
| 关节 jerk RMS (rad/s³) | 2134.523 | 619.128（降低71.0%） |
| 关节跟踪 RMSE (rad) | 0.338 | 0.096 |
| norm_return | 0.774 | 0.947 |

微调实际运行2000轮，比较250/500/750/1000/1500/2000轮后，按连续性选择1000轮，未简单选最后一轮。新旧参考按归一化相位比较，关节角 RMSE 约0.0145 rad，主要动作保留。
同一选中检查点在 mjwarp 上也完成316步，无失败，norm_return0.951；CPU全片段 eager/TorchScript 的316帧 qpos 最大差异为0。

- `artifacts/smooth_policy_only.mp4`：50 fps，无参考鬼影的真实策略演示。
- `artifacts/smooth_before_after.mp4`：新旧对照视频，按动作相位对齐，画面标注各自原生仿真时长；数值指标仍按原生50Hz轨迹计算。
- `artifacts/smooth_comparison.json`：基线、选中模型、跨后端结果和候选检查点的实测记录。
- `artifacts/smooth_final/`：NPZ起点的完整流水线产物，含模型、元数据、manifest、评估和视频。
- `artifacts/smooth_video_pipeline/`：本地MP4起点，实际重跑GVHMR/GMR及后续所有阶段的完整产物；没有跳过上游推理。
- `artifacts/smooth_mjwarp/`：同一检查点的 mjwarp 报告和视频。

已通过150项非slow测试，2项跳过、6项slow未选中。此结果针对当前非跳跃单人片段；跳跃接触模式需要单独验证。代码和训练/验证参数均保留，旧版产物不变。

## 其他舞蹈与新项目目录

`dance_projects/BV1ct4y1n7t9/` 是第二个独立训练项目，保存所选全身表演、动作数据、新训练和最终策略。参见该目录 README。

视频入口现先做按时间戳裁剪和 CFR30 归一化，避免 VFR/60fps 输入被直接当30fps；通过实际解码检查时长、帧数、空片段及越界范围，不只检查FFmpeg退出码。
SMPL-X在Y-up空间贴地，再由GMR转为G1的Z-up。实现放在当前repo的 `demo/retarget_gvhmr.py`，不再调用有轴错误的外部旧脚本；转换结果及元数据均保护已有文件。

```bash
# 从原始本地视频选择完整全身表演，独立训练另一份新run。
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python demo/pipeline.py \
  --video dance_projects/BV1ct4y1n7t9/inputs/source.mp4 \
  --source-start 13.4 --source-end 30.2 --camera static --flip-test \
  --contact-mode support --smooth-seconds 0.08 --slowdown 1.25 \
  --name ji_ni_tai_mei --checkpoint runs/demo_kick_smooth_001/ckpt_001000.pt \
  --train-iterations 4000 --backend mujoco --device cpu \
  --output-dir dance_projects/BV1ct4y1n7t9/reproduce_new_training
```

`--flip-test`对原始/镜像预测做平均，增加一次特征提取；`--camera simplevo`只估计相机旋转，不能保证修复所有推拉变焦/平移；当前新舞蹈选段已通过叠加检查。
讲话、镜头切换、多人和缺少腿脚的特写应先筛选；不能把单目不可见的腿部当成可靠训练标签。原视频与时间范围在每个项目的 `source.json` 记录。

相机选型必须对照背景验证：本次BV1ct4y1n7t9白色背景中，simplevo曾把舞者转身误判为相机旋转（诊断最高177°），虽能训练出稳定高分策略，转身朝向却不符。改用static后诊断降到约11°并与源片背身段对齐，另开run_002微调；最终交付只使用校正后的数据和模型。项目audit_camera.py可复现诊断，评分不能替代源画面对照。
