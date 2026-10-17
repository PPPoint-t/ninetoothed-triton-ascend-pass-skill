# NineToothed Triton-Ascend pass 迁移 skill

这是一个个人使用的本地 skill 仓库：帮助 agent 逐条理解 NineToothed 0.25 的 Ascend 适配规则，将有必要的部分落到 0.26 架构，并自动执行本项目算子回归、保存中文记录和恢复中断。

主流程采用单主控 agent、显式状态机、持久检查点、pytest 回归门禁、设备互斥和故障熔断。使用 Python 标准库实现本地控制器，不要求额外部署服务。可靠性以实际验证结果为准，见 [验收记录](验收记录.md)。

## 使用边界

- 源仓库：`/home/swy/infini/ninetoothed-swy`，默认 `dev-ascend`，只读。
- 目标仓库：`/home/swy/infini/ninetoothed`，默认 `dev-triton-ascend-backend`，保留原有未提交修改。
- 规范依赖：`/home/swy/infini/ninetoothed.skill`，只读，不把新 skill 写入该仓库。
- 算子测试只取目标 `tests/` 清单；不调度 ntops，不执行整个编译器框架/AOT 测试集。
- 当前 Ascend 保持 JIT-only；不增加公共能力报告、诊断平台或外部 compiler 指纹。
- 脚本不会修改编译器实现、断言或精度阈值；语义迁移由使用本 skill 的 agent 完成。
- 不自动提交/推送、安装依赖、清全局缓存、重置设备或结束其他人的任务。

## 测试文件变更规则

工作流默认运行目标仓库已有的算子测试。新增 Triton-Ascend 测试文件不是常规推进手段：先把 backend 参数、设备 fixture 或必要的断言并入原有测试。只有原有测试无法表达独立的 source/JIT/runtime 或 Launch ABI 契约，并且用户明确进入最终专用测试阶段时，才创建一个聚焦文件；必须在中文记录中说明不能复用哪一个原测试以及新增文件的唯一覆盖范围。

当前阶段不创建 `test_triton_ascend_*.py`、lockstep 矩阵或共享 case 工厂。删除或暂缓专用测试后要同步清单，不能留下指向不存在文件的测试入口。专用测试只能补充原有算子回归，不能替代原有测试，也不能用 skip/xfail 隐藏编译、数值或设备失败。

当用户明确要求验证目标 `tests/` 尚未覆盖的具体算子时，可以从只读旧仓库迁移一个最小算子测试到目标 `tests/`。记录必须包含旧测试来源、为什么现有测试无法表达、覆盖的唯一语义和后续是否保留；这不等于开放新增测试矩阵，也不允许直接运行旧仓库测试。

## 每条 pass 的实际推进顺序

迁移前先在目标 `tests/` 搜索对应算子的语义入口、`ntl` 调用和设备 fixture。找到覆盖后先运行 0.26 原状测试；没有覆盖时，才从 `/home/swy/infini/ntops` 找对应 kernel/test，并迁移一个最小案例。ntops 是参考来源，不是第二套回归套件。

每个案例都分别记录源码/lowering、工具链 build 和 NPU runtime correctness。0.26 已通过就登记“无需迁移”；部分通过就按失败的 shape、dtype、layout、mask 或控制流拆分路径；完全失败才逐条对照 0.25 pass。旧 pass 必须拆成单个变换逐次迁移，每一步都跑同一案例和非目标核心回归，直到找到最小必要改动。不能把整个 0.25 文件直接套进 0.26。

公共层只有在现有 IR 或 emitter 无法表达一个后端无关契约时才修改；已能表达的语义优先放在 Ascend 私有 SSA pass。每个变体都保留中文记录和可比较的执行编号，避免把“偶然通过”“未执行”或“改成 skip”误判为修复。

以 attention 为例，先保留当前 0.26 已通过的边界，再从 `seq_len=1, causal=False` 的失败路径开始，分别验证尾块 mask、sentinel 和 `exp2` 后 mask 等最小变化；只有能证明某一步是必要且不影响其他分支，才保留对应私有改动。

## 后端私有代码位置

backend class、schedule 和注册放在 `backends/<backend>.py`；较大的 SSA 语义修正拆到同目录的 `<backend>_passes.py`。共享 `emitters/<language>.py` 只承载语言共性和少量已验证的 backend 路由参数，Ascend 专用数值改写必须留在 Ascend 私有 SSA pass。独立语法或 intrinsic 才新增私有 emitter，设备工具链和 runtime 则放 materializer。公共 compiler/pass/frontend 只保留后端无关契约。

对具体算子采用三步对照：移除改动建立基线，恢复最小共享映射区分缺失语言支持，再验证私有 SSA rewrite 是否消除设备编译或数值错误。每一步分别记录 source、build 和 runtime 结果。

## 文件组织

```text
SKILL.md                         agent 的入口约束与连续工作循环
references/迁移规则表.md           M01—M12 的问题与新版落点
references/算子回归清单.md         精确函数入口、分类和核心模式
references/编排与恢复约定.md       状态、证据、设备故障与恢复
scripts/workflow.py               命令入口
scripts/engine.py                 收集、执行、比较和规则门禁
scripts/pytest_bridge.py          不改生产文件的实际产物/调用观测
scripts/device_probe.py           受限时的 NPU 健康检查
scripts/mdio.py                   中文表格、原子写入与文件锁
tests/                           工作流自身的故障与回归自测
```

运行记录默认放在目标仓库 `agent/`。中文 Markdown 是主记录，XML、traceback、源码和 SSA 是附属证据。无需阅读英文 JSON。

## 环境与 skill 发现

控制器需要 Linux、Python 3.11+。真实算子测试使用当前环境已有的 pytest、NineToothed 及其依赖；目标项目的 conftest 也依赖 torch，因此源码测试同样需要该测试环境。NPU 运行还需要 torch-npu、Triton-Ascend 和健康设备。无需 `pip install` 本仓库；使用正确的 Python 直接运行脚本。

本机已经安装：`/home/swy/.codex/skills/ninetoothed-triton-ascend-pass-skill` 是指向本仓库的符号链接，当前会话的可用 skill 列表已能识别它。

在新环境中，先将仓库放到约定位置，再用符号链接安装到默认 Codex skill 目录：

```bash
mkdir -p ~/.codex/skills
ln -s /home/swy/infini/ninetoothed-triton-ascend-pass-skill \
  ~/.codex/skills/ninetoothed-triton-ascend-pass-skill
readlink -f ~/.codex/skills/ninetoothed-triton-ascend-pass-skill
```

本机无需重复执行安装命令。如果链接位置已经存在，`ln -s` 会报错；先核对来源，不加 `-f` 覆盖。其他机器需按实际仓库位置调整链接，并准备使用边界中列出的旧仓库、目标仓库和只读规范仓库；链接本身不会安装这些依赖。链接安装后一般在新会话发现：

```text
$ninetoothed-triton-ascend-pass-skill
从已有中文检查点恢复，继续迁移下一条规则；仅跑目标 tests/ 算子回归。
```

**调用时只写 `$ninetoothed-triton-ascend-pass-skill` 即可，不需要在前面再加 `$ninetoothed-compiler-development`。** 本 skill 已要求 agent 读取 `/home/swy/infini/ninetoothed.skill/` 下的通用开发、后端适配和 pass 规范；这些文件仍需存在。并列调用两者通常没有必要。

只 clone/init 一个 Git 仓库不等于自动启用 skill。也可在对话中明确提供本仓库 `SKILL.md` 的路径。该 skill 不在会话结束后自行启动模型；要无人值守启动模型，需要另行提供外层 agent 启动器与预算。

## 第一次运行

以下命令在本仓库根目录执行。选一个尚不存在的批次目录：

```bash
python scripts/workflow.py init \
  --output /home/swy/infini/ninetoothed/agent/昇腾迁移-我的新批次
```

需要其他路径时，可显式传 `--source`、`--target`、`--policy`。初始化记录提交、相关源码、原有工作树 diff、环境和中文规则清单；已有输出目录会拒绝覆盖。

为便于复制后续命令，设置本任务变量：

```bash
NT_PASS_BATCH=/home/swy/infini/ninetoothed/agent/昇腾迁移-我的新批次
python scripts/workflow.py begin --batch "$NT_PASS_BATCH" --rule M01
```

一次只允许一条“处理中”规则。M10 的旧 AOT 包装默认暂缓，不代表新版二进制 AOT 已有实现。

## 收集与运行核心基线

```bash
python scripts/workflow.py collect --batch "$NT_PASS_BATCH" \
  --layer 源码 --group 核心 --backend 两者
python scripts/workflow.py collect --batch "$NT_PASS_BATCH" \
  --layer 设备 --group 核心

python scripts/workflow.py run --batch "$NT_PASS_BATCH" \
  --plan "$NT_PASS_BATCH/计划-源码-核心-两者.md" \
  --phase 基线 --timeout 120 --budget 900
python scripts/workflow.py run --batch "$NT_PASS_BATCH" \
  --plan "$NT_PASS_BATCH/计划-设备-核心-triton-ascend.md" \
  --phase 基线 --device 0 --timeout 120 --budget 900
```

`collect` 只收集，不执行 kernel。两种源码 target 分开归因；设备主队列只接受 `triton-ascend`。每个设备案例都检查实际调用 handle 的 backend、NPU 输入和同步，不靠环境变量猜后端。

退出码约定：`run` 返回 0 表示本次所选案例全部通过，1 表示存在失败/跳过/未完成，2 表示配置或执行前检查等错误；详细原因看中文记录。已有算子基线失败可以继续做修改后对照，不能只用 shell 的 `&&` 把“有旧失败”当成“工作流坏了”。`compare` 返回 0 表示无新增回归，但仍可能包含明确列出的原有失败；`finish` 另外检查相关目标算子和核心覆盖证据。

每个案例一个新 pytest 进程，`--timeout` 限制单例等待，`--budget` 限制一次执行的总预算。达到预算的剩余案例列为阻塞，不能计为通过。原 pytest 插件自动加载和 addopts 被清空，避免 xdist 或双 `-q` 意外影响运行。

`--device` 是物理卡编号，子进程映射到逻辑 `npu:0`。先确保这张卡可独占使用；锁只能协调本工作流，不能阻止外部程序占卡。

设备子进程继承互斥锁；每个 kernel case 启动前写全局“执行中”标记。主控异常退出后，即使子进程尚未结束或结果未落盘，另一批次也不能直接接管该设备。

## 相关回归与完整回归

相关组有 `逐元素索引`、`归约循环`、`矩阵布局`、`融合数学`。例如只复查 M01 的 fmod：

```bash
python scripts/workflow.py collect --batch "$NT_PASS_BATCH" \
  --layer 源码 --group 融合数学 --backend 两者 --match '*libdevice-fmod*'
python scripts/workflow.py collect --batch "$NT_PASS_BATCH" \
  --layer 设备 --group 融合数学 --match '*libdevice-fmod*'
```

命令打印实际计划路径，带筛选摘要；后续 `run --plan` 使用这个真实路径。筛选后的计划不能替代固定核心回归。

阶段结束使用 `--group 完整` 收集/运行清单中的原有算子参数。当前清单优先验证已有 `tests/` 算子在显式 `triton-ascend` 路由下的设备行为；不是 tests/ 的全部算子、全部形状，更不是框架/AOT tests。源码双 target 对照和独立 Launch ABI 测试留到用户明确进入最终专用测试阶段后再加入。需要其他算子或边界时，先扩展已有测试的参数或入口并更新分类，再重新收集；不能先创建新的测试文件来填矩阵，除非用户明确要求迁移某个具体缺失算子。

当前设备清单先覆盖原有的逐元素、clone、masked-load/pad、归约、matmul、softmax 和 CANN `libdevice.fmod` 入口；attention 只登记了 `seq_len=1` 的边界节点用于保留已知 correctness failure。scan、dropout 等其余风险项在用户明确进入最终专用测试阶段前不自动加入；加入后的失败仍保持失败，设备故障则停止队列。首次使用先跑核心，不直接用完整集验证设备是否正常。

## 修改后比较与规则完成

agent 分析旧规则并决定是否需要改代码；已有覆盖可以不产生任何生产 patch。需要修改时，先保存本轮文件/补丁，遵守私有层最小修改原则。

对相同计划用 `--phase 修改后` 再运行，然后比较实际执行目录：

```bash
python scripts/workflow.py compare --batch "$NT_PASS_BATCH" \
  --before "$NT_PASS_BATCH/执行记录/执行-0001" \
  --after "$NT_PASS_BATCH/执行记录/执行-0003"
```

执行编号由运行顺序生成，上面的编号仅示意；以自己的实际记录为准。比较要求同输入、测试、环境、清单、设备与计划。源码可以变化，这才是比较的对象。测试/环境本身变了需要新基线，不能拿旧结果强行比较。

比较按每个节点区分保持通过、本轮修复、原有失败和新增回归。原通过变 skip/未执行、失败原因变化或节点丢失均不能算通过保持。CLI 的 `compare` 通过只说明“无新增回归”，不说明“原有失败清零”。

完成 M01 时传入实际回归记录路径：

```bash
python scripts/workflow.py finish --batch "$NT_PASS_BATCH" --rule M01 \
  --decision 已有覆盖 --reason '当前 CANN 路由已有实现，fmod 与核心回归在两层验证，未新增生产代码' \
  --comparisons /实际路径/源码核心比较.md /实际路径/设备核心比较.md \
                /实际路径/fmod源码比较.md /实际路径/fmod设备比较.md
```

`finish` 校验当前源码的核心与相关算子证据，并检查同一源码版本此前的相关执行；不能挑选重跑后的成功结果来隐藏先前新增失败。真正修改源码后的新版本可以重新验证。未通过就保留“处理中”，或用 `--decision 暂缓 --reason '具体问题'` 保留原因。不能仅靠一句“已完成”跳过门禁。每规则最多尝试三个不同源码版本，防止无限试错。

## 看中文记录与恢复

```bash
python scripts/workflow.py status --batch "$NT_PASS_BATCH"
python scripts/workflow.py recover --batch "$NT_PASS_BATCH"
```

`status` 只读取并呈现已落盘状态，运行队列持有写锁时也可查看；`recover` 会更新恢复记录，必须取得本批次写锁。

重点文件：

| 文件 | 作用 |
| --- | --- |
| `执行总览.md` | 规则进度、各层执行数量及记录位置 |
| `规则记录/M01.md` | 中文语义判断和当前结论，可在结构表之后加 `## 语义分析` |
| `执行记录/执行-xxxx/记录.md` | 逐例通过、失败、跳过、超时、阻塞 |
| `回归记录/*.md` | 每个节点修改前后对照，是否出现新回归 |
| `恢复检查点.md` | 哪些执行中断、哪些证据因版本变化失效 |
| `环境与源码快照.md` | 初始工作树、导入环境与版本依据 |

结构表格字段有固定格式，脚本严格解析；可添加自由叙述，不删除/重复字段。状态记录写入会保留二级标题之后的人工分析。总览可以重建，旧执行证据不覆盖。

源仓库、规范或目标 HEAD 变化时另开批次。测试/清单变化后重新 collect。工作流脚本、Triton 关键代码、记录到的 CANN 版本或相关环境变化，会使比较条件变化；更换 BishengIR/CANN 仍需按既定流程清理相应 cache 并重启进程。

## NPU 故障时怎么办

AICore timeout、507014、设备执行进程 signal/超时或健康检查失败，会写全局设备隔离标记，后续案例阻塞。重启 pytest 不等于恢复设备。先查看日志并在外部完成设备恢复，确认没有遗留工作，再执行：

```bash
python scripts/workflow.py recover-device --batch "$NT_PASS_BATCH" \
  --device 0 --after-external-recovery
```

该命令只做受限时的真实健康检查，不 reset 硬件。超时清理的 TERM/KILL 等待也有上限；若子进程仍未退出，记录 PID 和进程启动标识，恢复命令会拒绝接管这个仍活着的进程。通过检查后用新执行编号继续，旧失败/中断记录仍在。

全局设备标记在 `~/.cache/ninetoothed-pass-workflow/`。不要手工删标记绕过故障。进程崩溃在 kernel 提交与结果落盘之间时，无法保证“恰好执行一次”；恢复会保留结果未知，不自动重放。

## 工具自身检查

在本 skill 仓库运行：

```bash
python -m pytest -q
ruff check scripts tests
ruff format --check scripts tests
python /home/swy/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
```

这里验证的是执行器行为，包括锁、超时、信号、状态恢复和门禁；不计入编译器算子通过数，不主动制造真实 NPU hang。各项真实验证和限制记录在 [验收记录](验收记录.md)。

## 当前限制

- 语义迁移需要运行中的 agent；控制器不自己调用 LLM 或后台反复修改代码。
- 默认只选现有 `tests/` 算子入口，新增模式先并入原测试并明确扩展清单和分类映射；用户明确要求迁移具体缺失算子时可增加一个最小测试文件，并记录其来源和保留理由。
- lazy toolchain build 和 runtime 目前同进程，未提供独立 build-only 性能/正确性结论。
- 本机无 NVIDIA/TileLang runtime 验证环境；旧 Triton 的源码对照不代表它们运行回归已通过。
- 本地设备锁不等于全机调度器；健康小运算不等于证明所有内核都不会 hang。
- 这是按成熟持久工作流模式实现的个人单机工具，可靠性范围见实际自测与试运行记录。
