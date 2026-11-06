# 部署级 sim2sim 代码与产物验收

日期：2026-10-29。源码：`src/unisupermimic/deploy/`；测试：`tests/deploy/`。

## 代码验收

- 变更前基线：192 passed、2 skipped、6 slow deselected。
- 最终默认完整套件：227 passed、2 skipped、8 slow deselected。
- 新增的两个实际 SDK2/DDS slow 集成测试单独运行：2 passed。原有6项slow训练测试本次未运行。
- 新源码与测试目录 Ruff 通过，Git whitespace 检查通过。
- 新进程导入全部部署运行模块、加载 bundle 后，未导入 `unisupermimic.tasks.*`，确认运行路径不调用训练 MDP/环境。

## 独立审查与 RED→GREEN 回归

1. 仅比较 state_dict 无法识别 TorchScript 固化的错误 action_smoothing。新增 canonical forward 对比，测试历史/previous action/goal 均非零。
2. 仅绑定 CSV 哈希不能识别帧率 JSON 被修改。绑定已验收保留副本的 CSV/JSON 哈希，并核对完整参考帧数。
3. DDS 原本缺少 native 的参考高度失败判据。新增 tracking_failure，在50Hz跟踪边界统一检查；根高度仅供仿真验收，不进入策略。
4. forward 对比的 previous-action 夹具原本在全部历史帧相同，会漏掉错误历史索引。改为跨批次/时间变化的输入，并加入故意读取最旧 action 的错误图回归。

上述问题均先复现测试失败，再修复通过。额外回归覆盖：非零姿态/腰关节下的观测等价、一次性朝向对齐、首次历史复制、限幅后 applied raw action、NaN/错误四元数/时间倒序/模型异常锁定、关节映射、PD力矩限幅、物理步结束的IMU缓存同步、空tracking不伪造帧、原子JSON状态发布、只准本机lo/非零domain、CRC、tick回绕和看门狗锁定。

## 实际产物验收

- 当前推荐 bundle_v3 与测试 bundle_v2 的 policy.ts/reference.npz 字节哈希相同；v3只增强导出时校验。
- 官方模型 native：记录1048帧、50fps、20.96秒，完整解码到末帧。
- 官方模型 DDS：832步完成后下一步触发倾斜故障，记录833帧、50fps、16.66秒，完整解码到末帧。
- 站立进入失败时 tracking NPZ 为 `(0,36)`，没有伪造跟踪分数。
- DDS 实际接收3335条状态，发送833条策略命令；物理故障后锁定阻尼。发布进程退出注入则由100ms看门狗触发。
- 物理轨迹和关节目标有限、目标在记录限位内，物理力矩受限；所有 readiness 字段保持false。
- 验收结束没有遗留 DDS 仿真/策略进程。旧模型、动作和视频未覆盖。

## 尚未通过的物理验收

完整DDS舞蹈、站立进入、舞蹈结束回站立、零初速度和组合扰动均未通过。代码测试通过不表示机器人可部署；后续需要与部署模型/低层PD一致的平衡过渡和鲁棒性训练，再复测。
