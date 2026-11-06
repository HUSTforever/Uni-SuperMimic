# Full Episode Controlled Trial

**Goal:** 检验完整连续物理episode训练能否改善从开始到结束的稳定性。
**Authority:** 用户持续优化全流程的授权；不改变完整目标或任何门槛。
**Evidence / hypothesis:** 原max_episode_steps=500，即10秒。完整reference2038步，即40.76秒；起始帧0的episode不会连续到25～27秒HF难段。随机中途初始化能够覆盖该帧，却不等于经历前段累计物理偏差。这是可检验分布差异，不预先认定根因。

**One-variable design:** 新run018_clock_full_episode使用与run017_clock相同源012125、fast_integrated_base_v1、seed0、独立生成器、critic_clock=True、HF1/1、LR3e-6、measured20、focus950:1398、32×32、500轮/预热50；唯一改变max_episode_steps从500至2038。reference/策略初始权重/PD/控制周期/力矩限制不改。新的完整episode自然在参考末端终止，保留真实失败保护与随机初始化。

- [x] 唯一配置差为500→2038，初始/25/50的pi与源逐张量相同，最终WM与策略归一化不变。
- [x] 500轮、512000实际控制转移、446.84秒完成，18个actor保存点均评估；3个通过保真、17个通过快速回站，0个全项通过。
- [x] 最佳满足其他门槛者HF约0.009495rad，仍高于0.005；主代理从原始物理数据重算确认。证据FULL_EPISODE_VERIFICATION_20261102.json；不推广默认，完整目标active。尚无新候选可进入剩余全链验收。

不新增产品代码或依赖；仅给实验脚本增加固定arm映射与保存新配置产物。保留所有旧结果。无需镜像实现的单元测试，验证使用实际配置差异、权重身份与物理运行。
