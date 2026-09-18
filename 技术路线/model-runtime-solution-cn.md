# TileLang 性能分析：模型路径、运行时路径与联合分析

本文面向项目技术介绍和团队协作，说明我们需要取得哪些信息、如何组织为 TileSight 输入，以及 GPU 观测如何用于独立报告和联合重算。它总结当前实现及已约定的边界，不新增设计承诺，也不代替逐测例的精度验证。

代码基线：本项目 `b15b4ac`；官方 TileSight `48e4158459bee5df830ae4ab7dda541edaa3dc4d`。当前主要目标平台为 NVIDIA H200。

配套阅读：[TileSight 流水接口](tilesight-ir-and-pipeline-guide-cn.md)、[TileSight 缓存接口](tilesight-cache-guide-cn.md)、运行时组件使用方法（实现仓库中的 `runtime/README.md`）。

## 1. 我们要交付什么

分析对象是一个明确的 kernel 及其输入、配置，不是整个 Op.forward 的混合时间。一个 Op 可能调用多个 kernel，还可能包含 Torch 数据准备；测试工具负责找到并绑定目标，不能把这些执行混为一谈。

工具围绕 tile 级计算和访存描述，分析 kernel 时间、资源需求、流水、缓存与访问效率，并提供两条可对比的路径：

| 路径 | 依据 | 产物 |
|---|---|---|
| 模型路径 | Python/高层 TIR 语义、显式输入、官方架构和成本数据 | Phase 工作量与依赖、缓存和流量估计、流水及 latency |
| 运行时路径 | 精确目标 kernel 的 GPU 执行与 NCU 观测；实验工具另采 CUPTI | 实测统计、独立诊断、原始证据及测试条件 |
| 联合分析 | 同一源层模型，加上可对应的运行时执行条件和访存需求 | 保留原模型，生成采用观测后的新预测及对比 |

联合分析是两条路径的组合方式，不是第三套成本模型。运行时报告不要求先得到完整模型 latency；不能关联到模型的观测仍可独立报告。

当前报告的六类视图是：tile pipeline 瓶颈、跨层搬运与访问效率、缓存利用、计算与 IO 重叠不足、负载不均、模型预测差异。每项需要区分“模型判断”“观测线索”“证据不足”，不能把视图存在等同于所有问题均能准确归因。

## 2. 总体数据流

```text
Python kernel + 构建参数 / 必要输入
                    ↓
          取得高层 TIR / PrimFunc
                    ↓
     统一语义事实：结构、操作、值、访问、执行域
                    ↓
        推导工作量、依赖、阶段和执行环境
                    ↓
       官方缓存分析 → 分层流量 → 官方成本
                    ↓
      Phase Timing + DAG / Region + Launch
                    ↓
        TileSight 流水及完整时间分析 ─────→ 纯模型报告
                    ↑
                    │ 仅替换已匹配的执行条件和资源需求
                    │
目标 kernel → NCU → 结构化观测 → 身份与口径检查 ─→ 运行时报告
                    │
                    └──────────────────────→ 联合预测与对比报告
```

CUPTI 的 kernel activity 时间用于另行统计与对比，不作为 Phase 成本输入。NCU duration 和 CUPTI duration 分别保留，不合并成一个无来源的“实测值”。

## 3. 为什么从高层 TIR 入手

Python 提供可调用入口、构建参数及特化上下文。执行 kernel builder 后取得 TileLang 增强的高层 TIR，操作、循环、存储及表达式已经成为结构化对象，比解释任意 Python 源码稳定。

模型路径不读取 generated CUDA、PTX、SASS 或 nvcc 的寄存器和 shared-memory 分配产物，也不依赖 GPU 执行取得语义。这样将源层语义解释与具体设备编译产物分开，便于后续更换平台和后端。

这不等于承诺 TileLang/TVM 安装包没有 CUDA 库依赖；“分析不调用 nvcc”与“依赖可在任意无 CUDA 环境安装”是不同问题。运行时路径仍需真实编译和执行目标 kernel。

Python macro 展开后的操作可进入 TIR。我们以取得的实际 IR 为依据，不再把 Python 调用外观当作第二套操作事实。

### 3.1 高层 TIR 的打印形式与实际提取入口

`builder.get_tir(...)` 返回结构化 IR；选定其中的 `PrimFunc` 后，调用 `prim_func.script()` 可以将其打印成类似 Python 的 TVM Script。此时仍能看到 tile 级调用，例如：

```python
# 高层 TIR 打印形式的节选；省略外层函数、Buffer 声明和循环内其他操作。
for ko in T.serial(32, annotations={"num_stages": 3}):
    T.copy(
        T.region(A[by * 128, ko * 32], 1, 128, 32),
        T.region(A_shared[0, 0], 2, 128, 32),
    )
```

这里已经有循环次数、索引表达式和 tile 访问范围，但还没有把 copy 展开成最终机器指令。打印出的 `for` 也不一定都是时间循环：`T.thread_binding(..., thread="blockIdx.x")` 等表达空间执行轴，必须读取节点的 kind 和 thread binding 加以区分。

**我们的结构和操作提取入口是 PrimFunc 对象，不是对整份打印文本按行或用正则重新建树。** 同一个对象可以分成两条使用路径：

```text
builder.get_tir(...) → 选定 PrimFunc
                        ├─ .script() → 供人阅读的文本及辅助身份记录
                        └─ 遍历 .body 等字段 → 结构、表达式、操作和访问事实
```

例如循环直接读取 `For.loop_var`、`min`、`extent`、`kind`、`annotations`、`thread_binding` 和 `body`；调用直接读取 `Call.op` 与 `args`；数据对象读取 Buffer 的 shape、dtype、scope、strides 等。变量及其引用按 IR 对象关系处理，不靠变量名字猜用途。实际节点命名可能随 TileLang/TVM 版本变化，因此提取规则需要维护版本边界。

### 3.2 分支与嵌套循环怎样存在于节点中

嵌套不是记录在一段缩进字符串里，而是**父节点的字段引用子节点**。下面是用于解释结构的简化打印示意，假设 A、C、D 已声明；不是一个完整 GEMM：

```python
for i in T.serial(8):
    for j in T.serial(4):
        if A[i, j] > 0:
            C[i, j] = A[i, j]
            D[i, j] = 1
        else:
            C[i, j] = 0
```

对应的主要节点关系为：

```text
For：loop_var=i，min=0，extent=8
└─ body → For：loop_var=j，min=0，extent=4
          └─ body → IfThenElse
                    ├─ condition → 比较表达式：BufferLoad(A, [i,j]) > 0
                    ├─ then_case → SeqStmt
                    │              └─ seq → [BufferStore(C, ...),
                    │                         BufferStore(D, ...)]
                    └─ else_case → BufferStore(C, ..., value=0)
```

| 节点 | 怎样保存下层结构 |
|---|---|
| `For` | `body` 引用循环体；循环体可以再次是 For，因此自然表达循环套循环 |
| `IfThenElse` | `condition` 是条件表达式；`then_case`、`else_case` 分别引用两条分支，省略 else 时后者为空 |
| `SeqStmt` | `seq` 保存按程序顺序排列的多个语句节点，不是字符串列表 |
| `BufferStore` / `BufferLoad` | 保存 Buffer、索引表达式，以及写入值等；这些表达式还可以继续包含其他表达式节点 |

实际 TIR 还可能包含 block、属性、局部变量绑定等包装。单条语句不一定有 SeqStmt 包装；表达式中的条件选择也可能是 `Select` 等表达式节点，不能只检查 IfThenElse。

提取时递归进入子节点，并保留当前循环、条件和执行范围：进入内层循环时同时保留 i、j 的上下文；进入 then/else 时分别保留条件成立／不成立的约束，退出后恢复外层上下文。因此，取出两个分支的事实不等于把两边都计作必定执行。若条件依赖线程编号，还需区分不同线程集合的参与关系，不能当成整个 CTA 只选一边。

这一阶段先保留嵌套和符号关系，不会无条件把循环全部展开；后续值求解、工作量和依赖组件再根据参数及必要输入确定执行范围。上例的条件取决于 A 的内容：仅凭 shape 无法确定各分支的执行次数，需要按第 4.2 节的输入规则处理。结构提取本身不要求先知道条件的结果。

## 4. 统一提取的信息

| 类别 | 提取内容 | 后续消费者 |
|---|---|---|
| 入口与参数 | 函数、参数身份、特化配置、shape/dtype、显式标量和必要输入值 | 值求解、测试身份、模型/实测关联 |
| 程序结构 | 顺序、循环上下界/步长、嵌套、条件、空间并行轴 | 结构组合、执行次数、循环前后边界 |
| 操作语义 | 矩阵乘、算术、数学函数、转换、归约、复制、原子、同步等 | 工作量和资源类别 |
| 数据与存储 | Buffer 身份、形状、类型、存储空间、可见别名/视图/offset/stride | 数据量、访问身份和复用 |
| 访问关系 | 读写对象、索引表达式、有效范围、掩码、访问方式 | 数据依赖、缓存映射、有效字节 |
| 值与状态 | 常量、定义引用、赋值更新、输入读取、跨轮状态 | 分支、次数、地址及跨轮依赖 |
| 执行与同步域 | grid、线程、显式分组、参与条件、提交/等待、barrier 参数、缓冲槽及流水配置 | Actor、同步关系、缓冲复用及 Launch |
| 来源 | IR 事实和表达式标识，以及可获取的 Python 源位置 | 审计和问题定位 |

这一步是规范化：保留后续分析需要的语义，消除部分表面写法差异，但不声称任意等价程序都会被化成相同形式。不能解释的调用或结构保留明确原因，不能把未知调用当成零工作。

### 4.1 不按 kernel 类别解释

语义表以调用身份、操作及上下文为依据；普通结构由统一遍历和解释规则处理。矩阵乘在 GEMM 和 Attention 中共享其操作语义，额外的 softcap 等操作应增加实际工作和依赖，不依赖“匹配完整 Attention 模板”。

主要实现位置：

- `model/semantics/extract.py`：TIR 事实提取。
- `model/semantics/catalog/surfaces.py`：统一调用语义表；数学和 shuffle 规则从相应规则模块汇入。
- `model/values/`：表达式、输入和有限状态求值。
- `model/graph/`：访问关系、依赖、同步及执行实例。

新的调用支持要同时检查语义、工作量及后端消费，不能只注册函数名就声称获得完整 latency。

### 4.2 两种输入分析方式

1. shape、dtype、配置和参数已足以确定结构与工作量：直接分析。
2. 必须读取具体输入值才能确定相关量：通过 `input_values` 等正式接口显式提供。测试工具集中保存相应输入生成与绑定方式。

提供输入后仍无法确定所需信息，则报错或保留不支持，不依赖执行 Op.forward 来偷偷补齐，也不按变量名猜数值。

区分变量的使用目的：只影响地址身份、影响计算数据，或影响工作量/控制。无需求出每一个浮点中间值，但某个地址读取若控制循环次数或有效访问范围，就不能忽略其值依赖。

## 5. 从事实形成 Phase、缓存输入和流水输入

### 5.1 工作量与计数范围

矩阵乘提取操作维度、类型和累加方式；逐元素计算统计操作类别及求值次数；归约保留归约轴、长度与输出数；访存结合有效访问区域计算字节。

必须标注每项工作是每元素、每线程、每 tile、每轮还是整个 grid。已按 `T.Parallel` 元素数计量的工作不能再重复乘线程数；引用已计算的值不能重复计算其定义。当前表达式复用限定于已支持的纯表达式范围，不等价于任意跨操作编译优化。

### 5.2 依赖与阶段

从读写关系、操作结果、异步提交/完成、同步通知/等待、跨轮状态和缓冲区复用推导依赖。源码中的前后顺序不自动等价于“前一操作必须完全完成”。

Phase 汇集一个调度单元的工作及资源需求，依赖在图中表达。tile 操作与 Phase 不要求一一对应：可以拆分或组合，但必须保留原工作量、数据关系、执行范围及来源。不得为得到较好数值而临时改阶段划分。

显式 warp specialization 从线程谓词和同步结构解释，不从 GEMM/GQA 等名称推断。线程子集通知与整个 actor 的人数分开；当前子集同步支持不代表任意子集工作量和状态均已支持。

完整有限实例可以形成有限调度；满足周期表达条件后再形成周期分析。没有周期证明时不能伪造 II。独立串行 Region 不会被 TileSight 自动合并成跨 Region 流水，未展开的嵌套循环也有官方接口限制。

### 5.3 缓存分析

同一份访问事实产生：数据 tile 大小、工作坐标到 tile 身份的映射、工作网格、循环次数及遍历方式。提交 `CacheProblem` 后，官方模型计算复用距离、命中率及分层流量。

它是规则访问模型，不直接消费递归 Region 树或最终 Phase 起点。返回的请求数和流量是其代表统计范围，不能无条件当作整个 kernel 已展开的总量。前端需要用真实工作发生次数统一范围。

### 5.4 官方成本与执行环境

计算工作量绑定官方吞吐；访存需求绑定官方缓存与带宽结果，形成 Timing。多个资源的 service time 与 Phase 完成时间分别记录，不重复相加。

Launch 表达 grid、线程、cluster 和驻留条件。当前纯源路径按线程、warp 和 shared-memory 声明等估计容量；源层有明确 H200 WGMMA 选择证据时，使用 1 CTA/SM 的简化假设。它不是硬件定律，也不是精确寄存器估计。

Phase Timing、依赖、缓冲容量、必要资源顺序以及执行环境一起交给 TileSight。其搜索结果是输入模型及已探索候选内的结果，不保证实际编译实现达到该性能，也不保证有限搜索找到了全局最优。

## 6. 架构决策：为什么由 adapter 直接组装 DAG

### 6.1 我们选择的是哪一条接入路径

**当前正式 source 路径主要由 adapter 直接构造 TileSight 的后端 Phase 和 PeriodicDAG，再调用官方调度分析；不是先拼好官方前端 Phase、Actor 和 pipeline，再调用 `lower_periodic` 自动组图。** 官方前端示例用于解释 TileSight 的接口，不代表 adapter 必须经过同一套 builder。

```text
官方前端方式
前端 Phase + Actor + 循环声明
    → TileSight lower_periodic
    → PeriodicDAG
    → TileSight 调度分析

我们当前的方式
Python / 高层 TIR
    → 我们的语义事实、工作量、执行范围和依赖
    → 官方成本能力及缓存分析 → Timing
    → adapter 组装后端 Phase + PeriodicDAG
    → TileSight 调度分析
    → 有限执行时间及 grid / 驻留组合
```

我们绕过的是官方前端的声明与约束转换，不是官方成本能力或流水求解。内部语义阶段、官方前端 Phase、后端调度 Phase 是不同层次的对象，不能因为都叫“阶段”就认为调用链相同。

### 6.2 这样做的原因

1. **避免重复表达。** 从 TIR 提取后，我们已经拥有分支、嵌套、线程参与范围、数据访问和同步关系。再把它们包装成另一套 Actor、sequence、carry、Buffer 声明，然后重新转换成边，会增加一层需要维护和核对的映射。
2. **保留已解析出的访问范围与数据版本。** 官方前端的自动推导有明确边界；直接构图可以提交已经证明的关系，避免重新组织逻辑 Buffer 来适应其推导规则。
3. **直接审计调度输入。** 每个后端节点的工作与 Timing、每条边的来源及迭代距离，都可以与我们的语义事实逐项核对。代价不是消失了，而是约束转换的责任明确留在 adapter。

下面两例依据固定版本官方 `lowering.py` 的规则，说明两种接入方式的区别。**它们是接口限制的示意，不是在声称某个当前测例正好因这些限制失败，也不是证明我们已支持任意多 writer 或跨轮访问。**

### 6.3 例一：同一个 Buffer 有多个写入者

假设源层已证明以下两次写入不重叠：

```text
load_left  → 写 S 的左半部分 ─┐
                            ├→ compute 读取完整 S
load_right → 写 S 的右半部分 ─┘
```

如果通过官方前端将两个加载都登记为 `writes=(S,)`、计算登记为 `reads=(S,)`，自动数据流转换会因 S 有多个 writer 而报错。当前规则按 Buffer 身份查找 producer，不会在这里进一步证明两个写入区域互不重叠。

走官方前端，需要先将 S 组织成两个适当的逻辑 Buffer，并调整读写声明。直接构图时，若我们已经证明区域覆盖和读写关系，就可以提交：

```python
from tilesight.fused_op_pipeline_wave.periodic_schedule import (
    Dependency as ScheduleDependency,
)

dependencies = (
    ScheduleDependency("load_left", "compute"),
    ScheduleDependency("load_right", "compute"),
)
# 这些边放入 PeriodicDAG.dependencies。
# 默认 min_delay=None，表示等待源 Phase 完成。
```

这只是数据就绪的两条边；缓冲复用和其他必要约束仍需另外提供，不能因绕过多 writer 检查就省略正确性分析。

### 6.4 例二：消费上一轮产生的数据

假设逻辑迭代关系已确定为 `produce(i) → consume(i+1)`，直接在后端表示为：

```python
dependency = ScheduleDependency(
    source="produce",
    target="consume",
    iteration_distance=1,
    min_delay=None,
)
```

但若在官方前端将二者登记为同一 Buffer 的 writer 和 reader，自动 buffer 数据流会生成 distance=0 的 `produce(i) → consume(i)`。额外声明正确的跨轮 carry 不会自动删除这条同轮边；`Phase.at(...)` 的迭代位置也不会改变这项自动推导的距离。

走官方前端，需要调整逻辑数据版本或读写声明，避免多生成约束；直接构图则提交我们已确定的跨轮边。首轮数据的初始化来源和后续槽位复用仍需处理，单独一条 distance=1 的边不代表完整循环已建模。

这不是说官方前端无法表达复杂程序，而是它的便捷推导规则未必直接对应我们已经得到的细粒度访问事实。

### 6.5 职责边界与实现位置

| adapter 负责 | TileSight 负责 |
|---|---|
| 从 Python／IR 解释操作、结构、输入值及工作量 | 使用官方计算吞吐、缓存与带宽分析能力 |
| 将已确定的提交／完成关系、跨轮状态、缓冲复用转换为后端约束 | 根据提交的时间、资源及约束搜索合法周期流水 |
| 保证计量范围、Phase Timing 和依赖距离正确，保留来源 | 返回 II、阶段起点、资源顺序及搜索完整性等结果 |
| 准备有限循环边界及空间执行参数，组织官方调用 | 提供所支持的有限执行和 grid／驻留时间组合能力 |

直接接入并不会扩大 TileSight 后端本身的表达能力，也不意味着我们自行实现调度求解器。不能用删依赖、漏记缓冲复用或静默简化执行结构换取 latency 输出。

当前相关实现位于 `src/tilesight_for_tilelang/model/backends/tilesight/`：

- `source_pipeline.py`：直接构造官方后端 Phase、Dependency 和 PeriodicDAG。
- `source_pipeline_costs.py`：组织阶段成本与周期分析，调用官方 `schedule_periodic_dag`。
- `source_pipeline_grid.py`：对接官方 `model_native` 的 grid／驻留等时间组合。

官方前端限制的核对依据：[固定版本 lowering.py](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/lowering.py)，其中 `_add_buffer_flows` 要求一个 Buffer 至多一个语义 producer，并将自动读写流设为同轮；`_lower_dependencies` 将这些自动边与显式依赖、carry 等合并。

## 7. 运行时路径：采什么，如何独立报告

运行时组件选择一次精确 kernel launch，保存设备、输入/构建配置、目标符号、launch 标识及采集环境。NCU 负责性能计数与 duration，测试工具可另用 CUPTI 收集 kernel activity 时间。

必须记录：GPU UUID/架构/SM 数、工具版本、频率、预热、重放方式、cache/clock control、采集命令和原始文件。不能仅凭 kernel 名字相同就比较或回灌。

观测被规范化为包含值、单位、作用范围、指标来源和有效状态的记录。缺失、不支持、无效和采集失败不能填零。报告保留原始计数，百分比和派生值另行标注。

独立报告可呈现时间、资源利用、缓存统计、分层访存、理论与实际 occupancy、stall/barrier、SMEM wavefront/bank、local-memory 等线索。它不会把 NCU 的 kernel 汇总自动解释为实测 Phase 起止或具体 warp 操作耗时。

程序按结构化产物和退出状态处理采集结果，不通过自然语言日志猜测稳定错误类别。设备冲突或权限不足时明确停止，不自动改变权限、驱动或锁频。

## 8. 联合分析：明确允许回灌的内容

### 8.1 A 类：执行条件

| 观测 | 用途 |
|---|---|
| grid/block/cluster、GPU、SM 数 | 核对源模型与执行对象是否一致 |
| 各资源限制下理论 CTA/SM 上限 | 替换源层驻留估计 |
| 寄存器分配、static/dynamic/allocated SMEM | 保留容量限制来源及诊断证据 |

理论容量与 achieved occupancy 不同。后者受执行进度、尾波等影响，不能直接换成整数驻留数。当前联合输入仅接受普通单 CTA cluster；源估计、观测容量和最终模型采用数分别保留。

### 8.2 B 类：可对应的访存需求

优先采用同口径、分方向的全路径 L2/DRAM 字节量，替换相应资源需求。直接字节量缺失时，当前支持从普通 LSU global-read 的 L1 hit/total sectors 派生 `miss sectors × 32` 字节，作为对应普通读取的 L2 请求需求。

该派生要求完整且唯一的 LSU 读取集合。不能把 TMA、未知 cp.async 路径或混合记录当成已证明的普通 LSU 读；不使用 NCU L1/TEX 命中率代替官方 L1.5，不把 L2 miss 直接当作全部 DRAM 流量。

同一资源/方向/范围只有一个采用来源。直接流量与命中计数派生量互斥，不重复回灌。

### 8.3 kernel 总量如何分到 Phase

NCU 通常没有我们所需的 Phase 粒度。联合路径采用公开的比例分配，不称为直接实测归因。

对于同一资源、方向和访问集合，设总观测字节为 B，第 i 类访问单次原需求为 b_i，完整 kernel 发生次数为 n_i：

```text
w_i = n_i × b_i / Σ(n_j × b_j)
分配给该类访问的总量 = B × w_i
单次采用量 = B × w_i / n_i
```

当前应依据匹配的原需求口径形成权重，不跨方向或混合路径分配。发生次数包含正确的 CTA、循环及 actor 次数；已经纳入 Work 的轴不能重乘。缓存代表请求数不一定等于这里的完整发生次数。

保存原需求、总量、权重、次数、采用量与 Phase 集合，并检查总量守恒。零权重、缺 owner、异质上下文或无法对应的额外流量不应强行分配。

不能排除 local/spill/atomic 混入、或与语义访问集合无法对应时，该全路径反馈可能被拒绝；保留原模型并报告原因，而不是把额外字节塞到访存最多的 Phase。

### 8.4 C 类：只报告，不直接回灌

| 数据 | 当前不直接回灌的原因 |
|---|---|
| kernel duration | 是对比目标，不能倒推 Phase 时间后再当作预测输入 |
| 利用率、IPC | 是执行结果，不直接变成吞吐折扣 |
| achieved occupancy | 不是资源允许的最大驻留容量 |
| stall/barrier 统计 | 不能自动转换成某条依赖的等待时间 |
| bank conflict、SMEM wavefront | 当前无已证明的 Phase 成本转换路径 |
| local-memory/spill | 归属、原因和额外依赖超出当前源模型范围 |
| 动态指令、原子和分支统计 | 尚不能与源 workload 无歧义逐项替换 |
| 实际频率 | 用于环境说明和比较资格，不自动重定标官方能力 |

### 8.5 正式反馈合同

```text
ObservedModelInputs
├─ source_digest / context_digest / observation_digest
├─ execution: ObservedExecution
├─ traffic: ObservedTraffic[]
└─ cache_counts: ObservedCacheCounts[]
```

类型集中在 model_observation.py（实现仓库中的 `src/tilesight_for_tilelang/contracts/model_observation.py`）。模型根包只消费这些数据，不依赖 runtime 包或 NCU 安装。

观测先经过身份和口径检查，再进入同一正式成本与调度链。联合结果保存实际被 native 消费的成本证据及摘要；不能为报告另算一套不同上下文的 Phase 成本。

## 9. 报告如何并列展示

报告分别保留纯模型、运行时观测和联合预测。每项数值明确标记来源：

- 源层事实或模型估计。
- 原始观测或单位转换后的观测。
- 按模型权重分配的观测需求。
- 显式假设、未采用反馈及原因。

kernel latency 可比较模型/联合预测与合格的实测记录；L2/DRAM 流量和理论驻留可按相同范围比较。Phase Timing、起点、II 与重叠仍是模型输出，NCU kernel 汇总无法提供对等实测值时应留空说明。

如果展示按总 kernel 时间折算的阶段指标，必须单列“模型比例折算”，不能命名为实测 Phase latency，也不允许再回灌成本。本文不宣称所有报告都已实现这一可选展示。

交付形式包括结构化 JSON、表单 CSV、HTML，以及已有调度数据的 Perfetto 展示。报告层只渲染和组织证据，不补造阶段起点、依赖或成本。

## 10. 一个矩阵乘示例贯穿两条路径

模型侧读取构建参数及高层 IR，得到加载 A/B、累加和写回、K 循环、读写区域及显式同步。分别推导计算量、缓存复用和流水依赖，绑定官方成本并得到纯模型时间。

运行时侧执行同一配置的 kernel，取得实际资源限制、分层计数和 kernel 时间。身份与计量范围匹配后，可以用理论 CTA 容量替换源估计，用合格的 L2/DRAM 总量调整对应访问需求，再调用同一 TileSight 路径得到联合时间。

整个过程不把实测总时间分给 load/MMA 后当成本输入，不按 Tensor 利用率乘一个性能系数，也不修改源码中的依赖来追随实测数字。若差异来自 lowering 的额外行为或未建模 spill，报告说明边界，不宣称联合路径已消除全部误差。

## 11. 这样划分的好处

1. **统一事实来源**：缓存、工作量和依赖共享变量身份、表达式与访问范围，减少重复解释和按名称猜测。
2. **TileLang 与 TileSight 解耦**：源层变化由提取及语义层处理，后端接口变化由 TileSight 适配层处理，报告消费稳定数据。
3. **模型与采集分离**：纯模型不要求 GPU 实测；runtime 可独立报告，联合分析是可选组合。
4. **成本来源明确**：使用官方已有成本数据，不在报告或采集层暗中引入第二套定标。
5. **可审计、可复算**：保留输入、来源、执行次数、选择与分配记录，能定位是语义、计量范围还是模型近似产生差异。
6. **测试能检查泛化而非记住算子**：用变量改名、等价表达、嵌套/互斥分支、不同参与人数及故意破坏同步等正反例，验证规则依据真实结构。此类对抗性构造是证据，不是对所有未来写法的完备性证明。

## 12. 组件边界及维护位置

| 目录 | 责任 |
|---|---|
| `src/tilesight_for_tilelang/contracts/` | 按语义、工作量、结果等子域集中维护类型和纯校验 |
| `src/tilesight_for_tilelang/model/` | 入口、语义、求值、图、Work 及官方 TileSight 接线 |
| `src/tilesight_for_tilelang/report/` | 数据组织与模型报告，不重新建模 |
| `runtime/` | 独立 NCU 采集、导入、诊断、报告及可选联合编排 |
| `tools/` | 固定测例表、源码版本、扫描、CUPTI 和实验编排 |

中心化不是把全部格式塞进一个巨大文件，而是每类合同有唯一权威定义和可查询来源。测试工具维护 Op/测例/kernel 的对应及 TileOps revision，但 Op 名称不进入通用语义判断。

当前仍需逐例记录语义或地址解释不完整、输入未绑定、复杂同步、非均匀执行和反馈范围不匹配等问题。本文不使用累计扫描通过数证明全仓精度，不承诺每项 runtime 指标均已回灌。

## 13. 接口与来源索引

- 架构与代码目录（实现仓库中的 `docs/architecture.md`）
- 语义映射维护（实现仓库中的 `docs/semantic-mapping.md`）
- 源层正式入口（实现仓库中的 `src/tilesight_for_tilelang/model/api.py`）
- 语义事实类型（实现仓库中的 `src/tilesight_for_tilelang/contracts/schema/semantic.py`）
- 驻留与完整时间接线（实现仓库中的 `src/tilesight_for_tilelang/model/backends/tilesight/source_pipeline_grid.py`）
- 访存成本接线（实现仓库中的 `src/tilesight_for_tilelang/model/backends/tilesight/memory_costs.py`）
- 观测反馈合同（实现仓库中的 `src/tilesight_for_tilelang/contracts/model_observation.py`）
- 联合分析实现（实现仓库中的 `runtime/src/tilesight_tilelang_runtime/joint.py`）
- 运行时使用与反馈边界（实现仓库中的 `runtime/README.md`）
- 报告说明（实现仓库中的 `docs/reporting.md`）

本报告通过当前源码、接口及相应文档交叉核对；不包含新一轮 GPU 测试或新精度结果。
