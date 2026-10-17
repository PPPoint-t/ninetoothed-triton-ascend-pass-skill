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
3. 先做算子发现，再做测试：在目标 `tests/` 中按语义名称、`ntl`/backend 调用、fixture 和 pytest node 搜索对应算子。若已有相关入口，先直接运行它建立 0.26 基线，不先读取旧 pass 就改代码；测试只选目标 `tests/` 下算子，不运行 ntops，不裸跑整个 tests/。
4. 若目标 `tests/` 没有能表达该语义的入口，才从只读 `/home/swy/infini/ntops` 查找对应 kernel 和测试，迁移一个最小、单一语义案例。记录来源、输入 shape/dtype/stride、设备、预期行为和唯一覆盖缺口；不能为了方便复制一组通用测试或创建 lockstep 矩阵。
5. 把基线按 source/lowering、toolchain build、runtime correctness 分层记录。根据结果做明确判断：0.26 已通过则记录“无需迁移”；部分通过则定位到具体 SSA op/def-use/shape/dtype 路径；完全失败才分析 0.25 pass。源码通过不能替代运行时正确，设备超时不能当普通算子失败。
6. 将 0.25 workaround 拆成最小可验证变换，每次只迁移一个变换并立即运行同一算子回归，再跑一组非目标核心回归。保留删除旧改动的 baseline、单变换 variant 和必要的组合 variant；找到最小能修复失败路径的版本后停止，不整体搬运旧 pass。
7. 必要实现放 Ascend 私有 pass/strategy/materializer。改动之前保存本轮相关文件和 patch，区别用户既有修改；公共层仅在有具体、可复现的后端无关扩展点阻塞证据时最小修改。前端、共享 emitter、SSA 公共拼接已经足够表达时，不为迁移方便增加新层。
8. 先检查 IR/源码和实际 pass trace，再执行相关算子与固定核心回归；阶段结束执行清单完整算子回归。每条规则最多验证三个不同源码版本，不无限修正。
9. `compare` 逐节点比较，再用 `finish` 写结论；新增失败、原通过变 skip/丢失、设备阻塞均不能完成。旧失败持续列为失败，部分通过必须保留已通过与失败子路径的差异。
10. 保存中文语义分析与证据链接，继续下一条。若设备熔断或语义无法证明，保留阻塞/暂缓原因；可以继续不依赖设备的分析，不能偷偷跳过失败。

## 算子优先、最小迁移门禁

每条 0.25 规则都必须绑定一个能观察其语义的目标算子和一个最小回归入口。工作顺序固定为：

```text
算子发现
 -> 目标 tests/ 已有测试基线
 -> 缺失时从 ntops 迁移一个最小案例
 -> source/build/runtime 分层
 -> 对照 0.25/0.26 SSA 与生成源码
 -> 单个最小变换
 -> 同一案例回归
 -> 非目标回归
 -> 证明最小必要 patch 或记录无需迁移/暂缓
```

“已有测试”按语义而不是文件名判断：要核对输入、shape、dtype、stride、mask、控制流和 backend 路由确实覆盖旧规则。只测试了同名函数但没有覆盖失败路径，仍属于覆盖不足；应扩展原有测试入口或在用户明确允许时迁移一个最小案例。

算子出现部分通过时，不得用全量旧 workaround 解释所有失败。先把失败缩小到一个可复现的路径，例如 attention 的 `seq_len=1, causal=False`、尾块 mask、有限负数 sentinel 或 `exp2` 后 mask；每个候选变换单独回归，并记录它是否改变了其他 shape、dtype、causal 分支。只保留能消除该失败且不引入非目标回归的最小变换。

迁移判断必须能回答三个问题：0.26 当前哪一个 SSA 结果已经正确；失败发生在 lowering、build 还是 runtime；0.25 规则中哪一个最小语义变换正好修复它。不能以“旧版本有这个文件”或“某个算子能通过”作为迁移理由。

## 测试文件门禁

- 默认复用目标仓库已有的 `tests/` 算子测试。先通过现有 fixture、参数化、backend helper 或测试入口表达验证，不创建新的 `test_triton_ascend_*.py`、lockstep 矩阵或共享 case 工厂。
- 只有在现有测试无法表达独立的 source/JIT/runtime 层、Launch ABI 或明确的后端契约，并且用户明确进入最终专用测试阶段时，才允许新增测试文件。新增前必须记录：为什么不能并入原测试、覆盖的唯一契约、预计保留期限和对应回归节点。
- 一个新能力优先追加到最接近的原有算子测试；不得因为想扩大矩阵、整理日志或重复验证 backend identity 而新建文件。重复的 lockstep、capability、materializer 文件应合并或暂缓。
- 删除或暂存专用测试后，必须同步更新 `references/算子回归清单.md`；清单不得引用不存在的测试入口。源码双 target 专用矩阵在最终阶段之前保持缺省，不为了满足门禁临时制造测试文件。
- 任何测试修改都不能把 correctness/build/timeout 失败改成 skip 或 xfail；设备不可用只能按设备层前置条件记录为阻塞。

用户明确要求验证目标仓库尚未覆盖的具体算子时，可以从只读的旧算子仓库迁移一个最小测试到目标 `tests/`，但必须在记录中写明来源、迁移原因、唯一覆盖范围和是否长期保留。这个例外只适用于具体算子语义回归，不能借此创建通用矩阵或 `test_triton_ascend_*.py` 测试集；仍然不运行旧仓库测试本身。

## 私有层归属规范

按下面的职责放置代码，先选择最窄的落点：

- `backends/<backend>.py`：backend class、该 backend 的 schedule policy 和 pass 注册。
- `backends/<backend>_passes.py`：当私有 SSA 语义重写或合法性分析超过一个小函数时使用；依据 opcode、类型、shape/layout 和 def-use，不按 kernel 名称匹配。
- `backends/emitters/<language>.py`：语言共性语法和已证明的少量 backend 参数化，例如 Triton 与 Triton-Ascend 的 libdevice import。这里不放 Ascend 数值 workaround。
- `backends/emitters/<backend>.py`：只有出现独立语法、intrinsic 或 ABI 时才新增；不能为了文件对称复制整个语言 emitter。
- `backends/materializers/<backend>.py`：私有 toolchain、设备、stream、JIT/AOT 和 runtime 行为。
- `compiler/passes.py`、frontend 和公共 IR：只增加后端无关 contract 或前端语义；没有具体阻塞证据不得加入 vendor 规则。

对具体算子采用三步对照：删除候选改动建立失败基线；恢复最小共享语言映射，判断是否只是 emitter 缺失；若已经能编译但设备数值错误，再把改写放入 Ascend 私有 SSA pass。每一步分别记录编译、运行和数值结果，不能把源码生成成功当成设备正确。前端只在错误参数会被静默丢弃时增加参数校验；如果当前算子没有证明该契约有必要，保持前端和共享层最小化。

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
