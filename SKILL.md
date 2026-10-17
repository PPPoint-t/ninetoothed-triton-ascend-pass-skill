---
name: ninetoothed-triton-ascend-pass-skill
description: 将 NineToothed 0.25 的 Ascend 专用规则迁移到 0.26 架构，使用本地规范、中文持久记录和 tests/ 算子回归连续推进。适用于旧 Ascend pass 继承、Triton-Ascend 私有层修改、迁移恢复及相关算子回归。
---

# NineToothed Triton-Ascend pass 迁移

## 先读取与定位

默认源仓库 `/home/swy/infini/ninetoothed-swy`，目标仓库 `/home/swy/infini/ninetoothed`；源只读，保护目标现存 dirty 修改。先检查真实分支、HEAD、diff 与导入路径，不执行 reset、stash、commit、push、安装依赖或设备 reset。

先读取 `/home/swy/infini/ninetoothed.skill/ninetoothed-compiler-development/SKILL.md`、`compiler-multibackend-porting/SKILL.md`、`compiler-pass-pipeline/SKILL.md` 及所需架构 reference。缺少规范时停止并解释，不跳过约束。用户的最小接入指示优先于通用建议。

读取本 skill 的 [迁移规则表](references/迁移规则表.md)，涉及执行/恢复时读取 [编排与恢复约定](references/编排与恢复约定.md)。脚本所在目录是本文件的 `scripts/`；可从安装链接解析真实路径，不依赖 shell 当前目录。

## 连续工作循环

1. 已有批次先 `recover`、`status`；无批次用 `init` 创建目标 `agent/` 下中文记录。命令见 [中文 README](README.md)。
2. 选下一条规则并 `begin`，以旧最终树为准，不重放已删除的历史 workaround。每次只处理一条。
3. 用清单 `collect`，运行修改前的相关案例和核心基线。测试只选目标 `tests/` 下算子；不运行 ntops，不裸跑整个 tests/。
4. 判断“已有覆盖、过时、需要私有实现、暂缓”。没有失败就不制造 patch。语义决定由当前 agent 完成，执行器不自动把 AST 替换成 SSA。
5. 必要实现放 Ascend 私有 pass/strategy/materializer。改动之前保存本轮相关文件和 patch，区别用户既有修改；公共层仅在有具体扩展点阻塞证据时最小修改。
6. 先检查 IR/源码，再执行相关算子与固定核心回归；阶段结束执行清单完整算子回归。每条规则最多验证三个不同源码版本，不无限修正。
7. `compare` 逐节点比较，再用 `finish` 写结论；新增失败、原通过变 skip/丢失、设备阻塞均不能完成。旧失败持续列为失败。
8. 保存中文语义分析与证据链接，继续下一条。若设备熔断或语义无法证明，保留阻塞/暂缓原因；可以继续不依赖设备的分析，不能偷偷跳过失败。

## 不可破坏的编译边界

- `TritonAscendOptimizeSchedule` 与 Triton 私有策略独立，共用现有通用候选机制；共享 emitter 与无设备假设的 JIT/ABI primitives。
- Ascend 不 import NVIDIA CUDA AOT helper；当前保持 JIT-only。旧 `AscendAOTBackend` 保存 Python wrapper，不等于设备 binary AOT。
- 不新增公共 capability report/诊断平台/缓存指纹系统；进度与失败分类保留在本 skill 和 `agent/`。
- 变换依据 SSA op、类型、shape/layout、def-use；禁止 kernel/变量名分派、整算子模板、source 后处理和其他 backend fallback。
- 不整体把 FP64 降为 FP32；SDPA 的有限负数替代 `-inf` 必须重新复现并证明所有消费者语义。
- source-only 不初始化 NPU。每项 schedule/metadata 必须有真实 consumer；注册新 pass 后必须验证它实际进入 pipeline。
- 不放宽数值阈值或给 correctness failure 增加 skip/xfail；不把源码通过计成工具链或设备通过。

## 执行与交付

NPU 默认一次一个独占通道，执行器检查实际 handle 的 backend 与输入设备。timeout/507014/进程信号使设备隔离；进程退出不是硬件恢复。恢复命令只在外部恢复完成后使用，不自动清理缓存/重置硬件。

主记录是中文 Markdown：配置、规则、执行、逐例回归和恢复检查点。原始 XML/traceback/生成源码只作附属证据。汇报真实命令、覆盖范围、未测层与失败，不把没有 NVIDIA runtime 环境写成旧后端全通过。

更换 BishengIR/CANN 后沿用用户政策：清理相关 Triton cache 并重启进程；不能为此扩展公共 cache。版本或测试变化后重新建立可比基线。

结束时运行 skill 自身验证与相关检查，生成中文交付报告。脚本提供状态机与执行，不在会话结束后自行启动模型；连续语义迁移需要有正在执行本 skill 的 agent。
