# 训练期动作越界正则

完整连续性目标及所有验收门槛保持不变。run018完整episode500轮的18个候选仍无全项通过；dq/gyro因果低通的四个变体也无全项通过，因此不加入部署。固定源策略的完整重放与旧qpos逐状态差0；在25～27秒窗口，左踝roll约54.5%的提议超出机械边界，其中约32.7%超出当前探索std的3倍，右踝pitch约7.9%超出5倍，最远约17倍。整个舞蹈中个别踝提议超出约24倍std。这是可观测的训练死区风险，尚不是已经证明的主要抖动根因。

## Approaches / Codex选择

增大探索方差才能越过很深的裁剪区，但会对正常关节加入更大扰动；局部无约束监督拟合可能破坏全身平衡，之前腕部实验已有此现象。本次选择在原物理PPO中加入可选、可微的均值越界项，让实际物理回报与信赖域检查继续约束全身动作。用户已授权持续优化并免去重复确认；以上算法取舍由Codex作出，不冒称用户指定。

## 数学与兼容性

配置action_bound_weight默认为0，有限且非负。物理raw边界由可信bundle计算：(joint_limit-default_q-context_offset)/action_scale；当前路径仍要求offset0。策略均值是经过原action_smoothing后的真实高斯中心，正则不作用于采样latent或已裁剪动作。

L_bound=mean(((relu(lower-mean)+relu(mean-upper))/action_std)^2)。包含边界的可行区内损失及梯度为0，越界梯度指回可行区。actor总loss为原PPO+原anchor+weight*L_bound。critic-only预热不计算或更新actor；权重0的分支不加入计算图，原采样RNG/critic/归一化/PI结构不变。输出仍raw29，原clamp保护、PD/力矩限制/参考/视频记录不改变。保存配置并日志action_bound_loss，明确该项只是训练正则。

## 验证与实际试验

解析损失与梯度/可行边界0/单次梯度步降低越界/非法std与bounds/配置非法项先RED后GREEN。真实collector测试强制一个actor输出越界，在critic-only阶段保持actor，启用actor更新时确有正则梯度且归一化/WM保持，保存与原导出契约通过；另验证weight0保持既有行为。完整常规suite与一次fresh review后，才运行有界500轮试验。先比较weight1.0，其他配置与run018完全相同；源仍012125，完整episode2038。18个actor保存点全部按原生HF/原始保真/快速回站验收；只有合格候选才进入完整源引擎/DDS/扰动/视频验证。

不承诺正则会使动作通过；若无改善保留证据与原默认，继续完整目标。
