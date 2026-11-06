# BV1ct4y1n7t9「只因你太美」→ G1 29-DoF

新舞蹈已在此独立项目完成训练、相机修正、两个物理后端验收和策略导出。旧「浪子踢球」项目保留。

**当前有效模型：`run_002/ckpt_001000.pt`；最终视频和导出：`artifacts/final_static/`。**

本目录保留原始舞蹈交付。后续面向官方模型/显式PD/延迟的微调实验位于 [g1_deployment_20261029](../../finetune_projects/g1_deployment_20261029/README.md)，新候选策略DDS跟踪3/3次完成，尚未通过起停过渡验收；没有覆盖本目录旧模型。

最新的 [完整连续起停版本](../../finetune_projects/g1_continuous_20261029/README.md) 已通过实际DDS全流程验收，并进一步降低抖动；旧舞蹈交付与之前失败证据均保留。

来源：https://www.bilibili.com/video/BV1ct4y1n7t9/

## 实际训练内容

完整原视频保留在`inputs/source.mp4`。开场为讲话，后半部分逐渐裁掉腿脚；本次使用13.4–30.2秒的完整可恢复全身表演，包含运球手势、转身、肩部动作和舞步，不只截取几秒容易的动作。
公开可取得的视频流为852×480、变帧率，标题“4K60FPS”不是本次下载规格。按真实时间归一化为30fps后，选段504帧、16.8秒；平滑并以0.8倍源节奏处理后，CSV630帧@30Hz，控制参考1049帧@50Hz，实际完整评估1048步、约20.96秒。

G1模仿身体和手臂手势，不包含篮球抓握、球体交互或真机控制。本项目用显式support接触模式，假设至少一脚支撑，小幅腾空稳定为支撑步态；主要姿态、手势和转身保留。

## 相机验收与训练

第一轮simplevo估计在近乎固定的背景中产生最高177°的虚假相机旋转。虽然策略能得到高分并稳定运行，转身朝向却不符；这一轮**不作最终交付**。
改用static+flip-test后，相机相对旋转诊断降至约11°、中位约3.4°；直接对照原视频25.4秒与G1模拟15秒，两者均为背身段。评分不能代替源画面对照。

训练先在run_001做4000轮身体动作/物理学习，再保留该试验并在run_002用正确固定相机数据独立微调1000轮。run_002重新建立优化器、配置和日志，比较250/500/750/1000轮后选用1000轮，完整跟踪回报最佳，连续性与较早检查点相近。

| 指标（当前正确相机数据） | 未适配旧策略 | 最终策略 |
|---|---:|---:|
| MuJoCo完成步数 | 591/1048，失败 | 1048/1048，无失败 |
| MuJoCo norm_return | 0.224 | 0.8900 |
| 完整新动作关节跟踪RMSE | 未完成整段 | 0.1250 rad |
| 同一前591帧跟踪RMSE | 0.3529 rad | 0.1147 rad |
| mjwarp完成步数 | — | 1048/1048，无失败 |
| mjwarp norm_return / 跟踪RMSE | — | 0.8718 / 0.1256 rad |

TorchScript/eager在完整1048帧MuJoCo闭环校验通过，最大qpos差异为0。192项非slow测试通过，2项跳过、6项slow未选中；独立代码审查发现的问题均已补充回归并修复。所有统计、正确相机候选模型和元数据见`RESULTS.json`。norm_return不是动作准确率百分比；被弃用的simplevo高分不能与正确数据的分数直接比较。

## 主要文件

- `source.json`：来源、源文件哈希、时间范围、实际帧率、已接受的相机/贴地/平滑参数。
- `RESULTS.json`：当前正式交付结果；`RESULTS_simplevo_rejected.json`保留错误相机试验供审计。
- `clips/ji_ni_tai_mei_fullbody.mp4`：原始选定全身片段。
- `upstream_static/ji_ni_tai_mei_fullbody/`：当前GVHMR预测和人体叠加视频。
- `upstream_static/g1_qpos_yground.npz`：正确Y-up人体贴地后的G1原始qpos。
- `motion_static/ji_ni_tai_mei_static.csv`与`.json`：当前36列训练参考及内容/帧率元数据。
- `run_002/`：当前微调的检查点、配置、训练日志；`run_001/`为先前试验，保留但不作为最终交付。
- `artifacts/final_static/policy_only.mp4`：50fps真实策略仿真，无参考鬼影。
- `artifacts/final_static/source_vs_policy.mp4`：源动作与G1对照，标明0.8倍源节奏。
- `artifacts/final_static/pipeline/policy.ts`及`policy.ts.json`：最终模型与raw action接口元数据。
- `artifacts/final_static/pipeline/manifest.json`：转换、评估、导出和整段验证记录。
- `artifacts/final_static/evaluation_mjwarp.json`：同一模型的GPU/mjwarp验收。
- `reproduction_static/manifest.json`：从原始VFR视频重新裁剪、CFR30、GVHMR、GMR直到完整闭环验证，全部7阶段实跑成功。
- `artifacts/reference_kinematic_static.mp4`：当前参考的运动学检查，明确不是策略rollout。
- `audit_camera.py`、`notes/camera_static_audit.json`：可复现的相机旋转诊断。

`upstream/`、`motion/`、`artifacts/final/`及`reproduction_check_1000/`是已弃用的simplevo试验，原样保留；不要误选其中模型或视频。
原始媒体、中间数据和大模型产物留在本地，不加入Git。上游GVHMR/GMR、人体模型及环境仍复用相邻的mocap目录，没有迁移或删除。

## 继续训练或复现

在supermimic根目录执行。训练入口指向当前正确固定相机数据，并拒绝覆盖已有运行：

```bash
bash dance_projects/BV1ct4y1n7t9/train.sh run_003
```

这一入口等价于`train_static.sh`，从run_001的身体动作权重出发，重新微调1000轮；具体参数完整保存在脚本和run配置中。旧错误相机试验的原始启动脚本单独留在`train_simplevo_trial.sh`，不会被默认调用。

从原始视频直接重新建立另一份正确相机的新训练，可用统一入口；每次选新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 \
.venv/bin/python demo/pipeline.py \
  --video dance_projects/BV1ct4y1n7t9/inputs/source.mp4 \
  --source-start 13.4 --source-end 30.2 --camera static --flip-test \
  --contact-mode support --smooth-seconds 0.08 --slowdown 1.25 \
  --name ji_ni_tai_mei_static --checkpoint runs/demo_kick_smooth_001/ckpt_001000.pt \
  --train-iterations 4000 --backend mujoco --device cpu \
  --output-dir dance_projects/BV1ct4y1n7t9/reproduce_new_training
```

仅重新导出/评估，用`--npz dance_projects/BV1ct4y1n7t9/upstream_static/g1_qpos_yground.npz`和`--checkpoint dance_projects/BV1ct4y1n7t9/run_002/ckpt_001000.pt`，训练轮数保持0，选择新输出目录，预处理参数保持上面一致。

## 流水线改进

- 按解码时间戳裁剪/CFR30归一化，检查实际帧数和覆盖范围，拒绝越界或空片段；不只相信FFmpeg退出码或可能错误的VFR容器duration。
- 修复SMPL-X Y-up贴地，然后由GMR转Z-up；显式记录源fps和校正轴。
- static/simplevo、flip-test与support/preserve-flight显式可选。保留腾空的固定地面模式不自动消除单目高度漂移，需要独立校验。
- NPZ和JSON都保护已有文件；通过独占文件句柄写NPZ，避免扩展名追加和误覆盖。
- 并行渲染测试只验证自己的临时XML清理，不把另一次渲染的文件误当作泄漏。

新输入仍需检查单人/腿脚可见性、真实背景运动和接触。此项目证明另一段舞蹈已可在G1 29-DoF仿真训练和运行，不表示任意剪辑或遮挡都能自动可靠恢复。
