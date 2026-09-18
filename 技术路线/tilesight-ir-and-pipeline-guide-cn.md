# TileSight 的阶段、前端接口与流水分析

本文用于项目技术介绍：先说明 TileSight 消费什么，再解释我们的前端需要生成什么。不展开 TileSight 内部的调度求解算法。

接口依据：官方 TileSight 固定版本 `48e4158459bee5df830ae4ab7dda541edaa3dc4d`。本文描述该版本，不承诺后续版本接口不变。

## 1. 分析范围与两个核心概念：Phase、PeriodicDAG

**PeriodicDAG 是 TileSight 周期流水分析真正消费的对象。TileSight 提供一套前端辅助组装工具，也支持用户自行构造这张图，再交给同一个调度器分析。** 图中的节点称为 Phase，表示一项 tile 级工作；边表示 Phase 之间的依赖，包括同轮与跨轮依赖。除了节点和依赖边，周期 DAG 还携带缓冲容量、必要资源顺序等约束。

我们输入的是 tile 操作的属性、时间和资源需求，以及操作之间的关系。官方前端将这些声明组装为 `PeriodicDAG`，调度分析器再求解阶段起点、稳态 II 和重叠关系；也可以直接提交已构造的 PeriodicDAG，跳过官方前端组装。**图是排程输入，不是已经排好的流水。** 循环次数、前后操作及 grid 等外层信息进一步用于组合完整执行时间。

我们的 adapter 负责从 Python／IR 提取上述信息，TileSight 不再重新理解原始 kernel 代码。

两条构图路径汇入同一个分析入口：

```text
方式一：使用官方前端辅助工具           方式二：用户自行构图
前端 Phase、Actor、循环关系           后端 Phase、Dependency、TokenBuffer 等
             │                                    │
             ▼                                    │
     官方 lower_periodic                          │
     解析 Timing、转换约束                        │
             │                                    │
             └─────────────────┬──────────────────┘
                               ▼
               PeriodicDAG：真正的周期流水分析输入
├─ 节点：后端 Phase
│        完成延迟 + 各资源占用时间与偏移 + 迭代位置
├─ 边：数据依赖、程序顺序、跨轮依赖
└─ 其他约束：缓冲容量、固定资源使用顺序
                               │
                               ▼
                 schedule_periodic_dag(dag, ...)
                 搜索满足约束的周期流水
                               │
                               ▼
                 II、阶段起点、资源顺序、重叠关系
```

**用户“自行构图”是提供节点及约束，不是预先排好各 Phase 的开始时间。** 官方前端不是调用调度器的必经入口；本项目当前主要采用右侧路径，具体原因见[模型方案第六节](model-runtime-solution-cn.md#6-架构决策为什么由-adapter-直接组装-dag)。

前端 Phase 的 Timing 可以显式提供，也可以由 oracle 解析；后端 Phase 必须已有完成延迟和资源需求。Actor 中明确声明的顺序会转成依赖边，但**资源竞争和缓冲容量并不全都转成普通 dependency 边**，还由资源占用和 TokenBuffer 等约束表达。搜索结果需结合搜索完整性理解，不等于实测执行轨迹。

**Region 的循环次数、前后操作，以及 Launch 的 grid、驻留条件，用于进一步将流水结果组合成完整 kernel 时间。** 它们属于外层执行组织与时间组合，不是上图中的 Phase 成员，也不是都被改写成 DAG 的边。

这里有两个最重要的概念：

- **Phase（阶段）**：表示一项 tile 级工作的基本流水单元，例如 tile 加载、矩阵乘累加或一组逐元素计算。
- **PeriodicDAG（周期依赖图）**：承载循环流水关系，包含阶段、同轮／跨轮依赖、缓冲容量和必要资源顺序，是周期调度器直接分析和计算的对象。

范围上需要区分：**一次周期 DAG 分析针对一个循环的重复工作及其跨轮关系；循环前后的顺序操作由 Region 层组合，不是自动加入同一个周期 DAG 排程。** TileSight 也能表达多个循环和嵌套 Region，但不会自动把整个 kernel 的所有循环合并成一个跨区域流水。嵌套周期宏的具体限制见后文“能力边界”。

### 1.1 前端 Phase：输入 tile 操作的信息

前端类型位于 `tilesight.tilesight_new_api.ir`。用下面的带注释调用形式看接口最直接；变量名代表待提供的数据，不是一段可独立运行的程序：

```python
Phase(
    name,           # 阶段名称
    actor,          # 所属执行角色
    owner,          # 所属循环的内部身份
    work,           # 工作内容及数量
    timing=None,    # 完成延迟和资源服务时间；未提供时需由成本入口解析
    reads=(),       # 读取哪些 Buffer
    writes=(),      # 写入哪些 Buffer
)

Work(
    kind,           # 工作种类，例如 copy、mma、pointwise
    flops=0.0,      # 计算工作量
    bytes=0.0,      # 搬运工作量
    attrs=(),       # dtype、形状等补充语义属性
)

Timing(
    latency,        # 阶段从开始到完成的时间，单位秒
    resources=(
        ResourceTiming(
            resource,       # 使用哪个建模资源
            service_time,   # 占用该资源多久，单位秒
            offset=0.0,     # 相对阶段起点何时开始占用
        ),
        # 可以继续列出其他资源
    ),
)
```

Phase 表达“这项操作做什么、做多少、读写什么、由谁负责、需要多少时间和资源”。**依赖是 Phase 之间的关系，在所属循环／pipeline 中另外声明，不是 Phase 构造参数里的内嵌字段。** grid、CTA 驻留等执行环境也在 Phase 之外。

通常通过已创建的 actor 登记 Phase：

```python
mma = consumer.phase(
    "mma",
    work=mma_work,
    timing=mma_timing,
    reads=(a_tile, b_tile),
    writes=(accumulator,),
)
```

这个接口会自动填写 `actor`、`owner` 并登记阶段。仅填写 Work 并不等于已经得到 Timing；`timing=None` 也不表示零耗时。

“tile 操作是基本单位”说的是建模粒度，不要求 Phase 与一个 TileLang 调用或一条机器指令一一对应。依据工作和调度边界，一个操作可以拆成多个 Phase，多个操作也可以组合为一个 Phase。

### 1.2 PeriodicDAG：循环流水关系的载体

将阶段及其约束组织起来，就得到周期调度器直接消费的 `PeriodicDAG`。固定版本的接口共有四组字段：

```python
PeriodicDAG(
    phases=phases,             # 后端 Phase 集合：阶段及其时间、资源需求
    dependencies=(),           # 同轮和跨轮依赖
    token_buffers=(),          # 有限容量的缓冲槽及获取／释放关系
    fixed_resource_orders=(),  # 必须保持的资源使用顺序
)
```

其中 `phases` 是必填参数，且不能是空集合；其余三组默认是空元组。这里不能把 `phases=()` 理解成合法的空图默认值。

**所以，把 PeriodicDAG 称为“循环流水关系的载体”是准确的。** 它不仅列出有哪些操作，还规定了操作何时具备开始条件、跨轮需要等待什么、缓冲槽是否足够，以及共享资源的必要顺序。Phase 中的资源需求则让调度器识别资源竞争，不需要为每一对竞争者手写依赖边。

DAG 描述允许怎样执行，不是已经排好的时间线。调用：

```python
result = schedule_periodic_dag(dag)
```

才会得到阶段起点、稳态 II、资源使用顺序和相关分析结果。循环次数及前后顺序操作不在这四组字段里；它们由 Region 等外层接口组织，再用于计算有限执行时间。

### 1.3 真正放进 DAG 的后端 Phase 和 Dependency

`PeriodicDAG.phases` 中的 Phase 不是 1.1 的前端对象，而是 `tilesight.fused_op_pipeline_wave.periodic_schedule.Phase`。流水层只保留排程需要的信息，接口更精简。为避免同名混淆，下面使用 `SchedulePhase` 和 `ScheduleDependency` 作为导入别名：

```python
from tilesight.fused_op_pipeline_wave.periodic_schedule import (
    Phase as SchedulePhase,
    Dependency as ScheduleDependency,
    ResourceUse,
)

SchedulePhase(
    name,                    # 当前 DAG 内唯一的阶段名称
    latency,                 # 阶段完成延迟，单位秒
    resources=(
        ResourceUse(
            resource,        # 与其他阶段竞争的建模资源
            service_time,    # 资源占用时间，单位秒
            offset=0.0,      # 资源占用相对阶段起点的偏移
        ),
    ),
    iteration_offset=0,      # 展开窗口中，该阶段属于哪个逻辑迭代
)

ScheduleDependency(
    source,                  # 源阶段名称
    target,                  # 目标阶段名称
    iteration_distance=0,    # 目标相对源跨越几轮；0 表示同轮
    min_delay=None,          # 默认等待源阶段完成，也可指定就绪延迟
    name="",                 # 可选的依赖名称
)
```

`iteration_offset` 不是时间或循环次数，也不是依赖边的 `iteration_distance`。前者标识阶段在展开窗口中的逻辑迭代位置；后者描述源、目标之间的跨轮关系。

后端 Phase 不再保存 `Work`、`actor`、`reads`、`writes`：工作已落实为 Timing，相关执行语义已转换成 DAG 约束。前后两层的核心映射是：

```text
前端 Phase.name                      → 后端 Phase.name
解析后的 Timing.latency              → 后端 Phase.latency
Timing.resources: ResourceTiming      → 后端 Phase.resources: ResourceUse
actor/resource 序列中的 at(...) 声明 → 后端 Phase.iteration_offset
依赖、跨轮状态、缓冲与必要顺序         → DAG 的其他三组字段
```

官方转换入口 `lower_periodic(kernel, loop, oracle=None)` 在构建 PeriodicDAG 时生成这些后端 Phase，再交给周期调度器。**后端 Phase 是排程输入；阶段起点和 II 是排程输出，不是 Phase 构造参数。** 同样，DAG 的 II 也不等于完整 kernel latency。

固定版本源码：[前端 Phase、Work、Timing](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/ir.py)、[前端声明接口](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/frontend.py)、[DAG 转换](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/lowering.py)、[后端 Phase 与 PeriodicDAG](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/fused_op_pipeline_wave/periodic_schedule.py)。

## 2. 从整体调用入口向内展开

TileSight 支持两种组图入口，最终都由同一个周期调度器分析：

| 用法 | 谁组装后端 DAG | 接入位置 |
|---|---|---|
| 使用官方前端声明 | TileSight 根据前端 Phase、Actor 和循环关系转换 | `PeriodicAxisIR(loop_name="k_pipeline")` |
| 用户自行构造 DAG | 用户直接创建后端 Phase、Dependency、TokenBuffer 等 | `PeriodicAxisIR(dag=dag)`；只分析流水时也可直接调用 `schedule_periodic_dag(dag)` |

`dag` 与 `loop_name` 二选一。**自组装 DAG 是官方支持的入口，不需要先把图重新包装成官方前端依赖，再让 TileSight 组装一次。**

### 2.1 完整区域与 kernel 时间分析入口

原生区域执行器的入口是：

```python
model_native(kernel, region, arch, options=None, oracle=None)
```

| 参数 | 职责 |
|---|---|
| `kernel: KernelIR` | kernel 声明，包括 launch、前端周期循环、actor、阶段及相关约束 |
| `region` | 本次分析的执行结构：阶段、顺序组合、循环及边界 |
| `arch` | GPU 的计算、存储和容量等架构能力；具体成本路径消费相应字段 |
| `options: NativeModelOptions` | II 模式、边界策略、多 CTA 组合等选择 |
| `oracle` | 可选成本解析入口；用已绑定的 Phase Timing 时可不提供 |

输出 `NativeModelResult` 包括 `region` 分析结果、`per_work_unit_s`、`resident_group_s`、`tail_wave_s`、`kernel_body_s`、grid/wave 信息、驻留信息和相关报告。`total_s` 还包含显式配置的 `launch_s` 与 `host_dispatch_s`，比较时间时必须区分口径。

整体对象关系如下，region 分支先以一个循环为例：

```text
model_native
│
├─ kernel: KernelIR
│   └─ LaunchIR
│       ├─ work_grid / physical_grid
│       ├─ threads / cluster / residency
│       └─ PeriodicLoopIR（按名称登记的循环流水声明）
│           ├─ ActorIR → 前端 Phase → Work、Timing、读写 buffer
│           ├─ dependencies / carries
│           └─ lifetimes / pipeline_buffers / resource_sequences
│
├─ region: 执行结构（传入一个根区域）
│   └─ LoopRegion
│       ├─ name / trip_count
│       ├─ prologue → 前置区域，例如 PhaseRegion
│       ├─ body     → 引用前端 Phase 的区域结构
│       ├─ epilogue → 后置区域，例如 PhaseRegion
│       └─ periodic_axis（可选）
│           ├─ dag = 显式 PeriodicDAG
│           └─ 或 loop_name = 引用 kernel 中的周期循环
│
├─ arch: 硬件能力
│
├─ options: 分析策略
│
└─ oracle: 可选的成本解析接口
```

**有多个 region 时，不是给 `model_native` 连续追加多个参数，而是先组织成一个根 region。** 如果它们按完成后再开始的顺序执行，就依次写进 `SequenceRegion.children`：

```python
region = SequenceRegion(
    name="main",
    children=(initialize_region, loop_region, store_region),
)
result = model_native(kernel, region, arch)
```

每个 child 可以是 PhaseRegion、LoopRegion 或另一个 SequenceRegion。这里的顺序组合明确要求子区域串行完成，不能用它冒充任意并行关系；同一组前后操作也不要既放在 children 中，又重复放进循环的 prologue/epilogue。

**`PeriodicLoopIR` 是循环流水的声明，不只是循环的名字或结构占位。** 它保存参与的 actor、Phase、依赖、跨轮状态和容量等约束；`LoopRegion` 则描述该循环在程序中的位置、执行次数、body 和前后操作。两者通过 `LoopRegion.periodic_axis.loop_name` 显式关联：例如区域可以叫 `k_loop`，而 `loop_name="k_pipeline"` 引用已登记的 `PeriodicLoopIR("k_pipeline", ...)`。两个对象自己的名称不必相同，也不会按名称相似性自动绑定。

使用 `periodic_axis.dag` 时则直接提供后端图，不必再通过 `loop_name` 转换；两种绑定方式二选一。循环外的前后顺序操作由 region 树表示，不是都登记成这个周期循环中的重复工作。

这些是建模对象，不是 TileLang/TVM 编译器中的原始 TIR 节点。

### 2.2 用户自组装 DAG：直接分析或接入 Region

下面用加载和计算两个节点展示接口位置。时间是教学输入，不代表实际性能；这里只展示局部关系，完整程序还应提供所需的累加状态等约束：

```python
from tilesight.fused_op_pipeline_wave.periodic_schedule import (
    Phase as SchedulePhase,
    Dependency as ScheduleDependency,
    ResourceUse, TokenBuffer, PeriodicDAG, schedule_periodic_dag,
)

load = SchedulePhase(
    name="load", latency=100e-9,
    resources=(ResourceUse("tma", service_time=60e-9),),
)
compute = SchedulePhase(
    name="compute", latency=180e-9,
    resources=(ResourceUse("tensor", service_time=180e-9),),
)
dag = PeriodicDAG(
    phases=(load, compute),
    dependencies=(
        ScheduleDependency("load", "compute"),  # 本轮加载完成后开始计算
    ),
    token_buffers=(
        TokenBuffer("tiles", "load", "compute", capacity=3),
    ),
)

# 用法一：只分析周期流水，不需要 KernelIR、Region 或 grid。
envelope = schedule_periodic_dag(dag)
```

如果还要计算循环前后操作和完整 kernel 时间，则把**输入图 `dag`**放进 Region 的 `periodic_axis`，不是把分析结果 `envelope` 放进去：

```python
from tilesight.tilesight_new_api.regions import LoopRegion, PeriodicAxisIR
from tilesight.tilesight_new_api.native_executor import model_native

# 接线示意：kernel_ir、iteration_region、前后区域和 arch 已提前构造。
root = LoopRegion(
    name="main_loop",
    trip_count=128,
    body=iteration_region,
    prologue=initialize_region,
    epilogue=store_region,
    periodic_axis=PeriodicAxisIR(
        dag=dag,                    # 用户构造的后端图；不再填写 loop_name
        ii_mode="periodic_best",
    ),
)
result = model_native(kernel=kernel_ir, region=root, arch=arch)
```

这两段展示的是两种调用用途，**完整分析不要求先执行一次 `schedule_periodic_dag`**。执行器取得显式 DAG 后，会自行调用周期分析。

需要注意：绕过前端组图，不等于绕过 `model_native` 的所有输入要求。当前 Region 的 `PhaseRegion` 仍引用前端 Phase；上面的 `iteration_region` 必须包含与 DAG 中 `load`、`compute` 的名称、完成时间和资源时间匹配的阶段，`kernel_ir` 仍提供 launch 等声明。后端 Phase 不能直接当作 PhaseRegion 的前端 Phase 使用。区别是周期依赖、缓冲和资源顺序以 `dag` 为输入，不再通过 `loop_name` 自动推导；可选的前端生命周期关联由 `liveness_loop_name` 另行指定。

本项目当前主要采用“自行构造后端 DAG，再调用官方分析”的方式；完整实际调用链及 CTA 汇总接线见第 11 节，不应将本节接口示例当成 adapter 已逐项采用的完整 Region 组织方式。

## 3. Region：程序如何组织与重复

当前公共 Region 类型主要有三种：

| 类型 | 关键字段 | 普通区域执行语义 |
|---|---|---|
| `PhaseRegion` | `phase`、可选 `label` | 一个工作叶子，不是区域入口标记 |
| `SequenceRegion` | `name`、`children` | 子区域按完成后再开始的顺序组合 |
| `LoopRegion` | `name`、`trip_count`、`body`、可选前后区域及周期配置 | 重复 body；未绑定周期配置时按串行重复分析 |

`SequenceRegion` 没有 `body` 或 `dependencies` 字段。`LoopRegion` 的 body、prologue 和 epilogue 都可以是其他 Region，因此可以递归嵌套。

`PhaseRegion(load_a)` 表示该叶子包含 load_a 这项工作。阶段的时间端点是 `load_a.start` 和 `load_a.done`，不由 PhaseRegion 表示。

### 3.1 为什么 Region 已有循环结构和次数，还需要周期 DAG？

“load、compute 重复 128 次”不能区分完全串行和跨轮预取。循环名称和次数不能回答数据何时就绪、是否存在累加器跨轮状态、缓冲槽何时释放。

因此，Region 描述程序的组织和次数，DAG 描述允许的执行方式。只把几个工作包装成 Sequence，不会得到任意 DAG 的并行关系。

在周期宏路径中，LoopRegion 的 body 用于匹配 Phase、成本和区域结构，显式 DAG 用于流水约束。官方示例允许 `body=SequenceRegion(...)` 同时绑定周期 DAG；此时不能把这个 body 的排列误读为强制所有迭代串行。周期执行器会检查 body 与 DAG 的节点名称、完成时间和资源时间一致，然后按周期策略计算循环时间。

### 3.2 PeriodicAxisIR 是循环与流水描述的关联，不是 TIR 变量

| 字段 | 用途 |
|---|---|
| `loop_name` | 引用 KernelIR 中的 PeriodicLoopIR，使用官方前端转换 |
| `dag` | 直接提供已构建的 PeriodicDAG；与 loop_name 二选一 |
| `liveness_loop_name` | 关联前端 buffer 生命周期声明；loop_name 路径默认关联同名循环 |
| `ii_mode` | 按循环指定分析模式；`inherit` 使用全局 options |
| `summary_policy` | `auto`、`macro`、`inline` 配置；当前 native 对周期 `inline` 明确报不支持 |
| `boundary_anchor_phase` | `finite_witness` 边界策略要求的 DAG 锚点名称 |
| `search_config` | 调度搜索配置 |
| `legacy_stage_boundary` | 旧式 stage 边界策略的附加配置，不是通用依赖提取规则 |

`summary_policy` 决定循环如何交给外层组合：`macro` 先分析内部周期 DAG，再将整个循环汇总为一个执行单元；当前固定版本的 `auto` 也走这条路径。`inline` 意图展开内部执行，但当前执行器对非零轮次的周期循环尚不支持，会报错。它不是最佳／最坏流水的选择（那是 `ii_mode`），也不能用来开启跨 Region 重叠。

`boundary_anchor_phase` **不是依赖边**，而是 `boundary_policy="finite_witness"` 下拆分时间的参照 Phase 名称。例如指定 `"mma"`：循环开始到第一轮 mma **开始**为启动段，第一轮到最后一轮 mma **开始**之间为中间段，最后一轮 mma 开始到循环内全部操作完成为收尾段。prologue、epilogue 再分别计入启动和收尾部分。更换有效锚点只改变分段口径，不增加依赖，也不改变这次有限调度的完整执行时间。

例如 TIR 循环变量 `k`、Region 名称 `k_loop`、前端循环名 `k_pipeline` 可以不同。前端建立关联，TileSight 不按名称相似性猜测对应关系。

## 4. Actor：执行角色，不是硬件资源本身

Actor 表示一组操作的执行角色。builder 中通过 `pipeline.actor(...)` 创建它，登记工作和顺序后，`kernel.build()` 将其保存为 `ActorIR`。下面是全部 7 个成员的调用形式；`actor_phases` 代表已登记的前端 Phase：

```python
ActorIR(
    name="producer",              # 角色名称，不按名称推断硬件行为
    serial_resource=None,         # 兼容旧接口的固定资源顺序声明，通常不填
    phases=actor_phases,           # 属于这个角色的前端 Phase 集合
    sequence=(),                  # 显式的周期操作顺序：Phase.at(...) 序列
    order="issue",                # sequence 的含义：issue 或 completion
    execution_scope="warpgroup",  # 角色的执行范围，不是资源名称
    execution_domain=None,        # 可选的执行范围数量等补充描述
)
```

| 成员 | 表示什么 | 不表示什么 |
|---|---|---|
| `name` | 当前循环内的角色名称，例如 producer、consumer | 不因名称叫 producer 就自动产生 TMA 操作 |
| `serial_resource` | 旧接口兼容字段；配合 actor sequence，生成对应资源的固定周期使用顺序 | 不是该 actor 的全部资源清单，也不提供资源耗时；新声明优先使用独立的 `pipeline.resource_sequence(...)` |
| `phases` | 这个角色负责的工作集合 | 登记顺序本身不要求前一个完成后才开始下一个 |
| `sequence` | 显式指定的 `Occurrence` 序列，例如 `(a.at(0), b.at(0))`；每项引用 Phase 和所在迭代位置 | 不是已算出的时间线；空序列表示未声明这类角色顺序，不表示没有其他依赖 |
| `order` | `issue` 生成开始到开始的零延迟约束；`completion` 生成完成到开始约束 | `issue` 不包含实际指令发射的耗时；没有 sequence 时，仅设置 order 不会产生顺序边 |
| `execution_scope` | `unspecified`、`thread`、`warp`、`warpgroup`、`cta` 或 `cta_group` | 不是具体线程编号，也不是 SM 上的 CTA 驻留数 |
| `execution_domain` | 可选 `ExecutionDomain`：补充 scope、每 CTA 的实例数、每实例成员数以及来源标记；scope 必须与 actor 一致 | 不自动给出物理线程映射，也不替代 Phase 的工作量和 Timing |

例如两个负责加载的 warp 可以用 `scope="warp", instances_per_cta=2` 描述其执行范围；这不是“每个 SM 驻留两个 CTA”。执行范围信息可供归属与生命周期等分析使用，不能仅凭这个数量就认为 DAG 自动复制了全部工作。

- actor 回答：哪些操作由同一执行角色负责，明确的程序顺序是什么。
- ResourceTiming 回答：操作执行时使用什么建模资源，使用多久。
- Dependency 回答：什么事件发生后，另一个事件才能发生。

同一 actor 发起的计算可以在不同资源上重叠，不同 actor 也可以竞争同一资源。actor 不自动模拟机器指令发射单元，也不是必须等同于具体物理线程编号。

仅把 Phase 登记到 actor，不会自动声明完成串行。显式调用：

```python
actor.sequence(a.at(0), b.at(0), order="issue")
```

表示提交顺序；`order="completion"` 则使用完成到开始的约束。此 sequence 是周期序列，官方转换会补末项到下一轮首项的关系。`at(...)` 的偏移表示操作在展开窗口中属于哪一轮，不自动改变其数据依赖距离。

相同资源名称的 ResourceTiming 表示对同一建模资源的竞争。通常不应为每一对竞争者手动加依赖；只有必须固定仲裁顺序时才声明 `resource_sequence`。不能仅凭 Python 源码行号把所有 actor 的操作固定成一个全局顺序。

## 5. 工作量、完成时间与资源占用

### 5.1 完成时间与资源占用不是相加关系

下面给出一个同时使用 CUDA 和 SFU 两种资源的 Phase Timing。**数值只是说明接口的教学输入，不代表某个实际操作的定标结果。**

```python
ns = 1e-9
timing = Timing(
    latency=100 * ns,  # 整个阶段从开始到结果可用，共 100 ns
    resources=(
        ResourceTiming(
            resource="cuda", service_time=60 * ns, offset=0 * ns,
        ),  # 相对阶段起点，在 [0, 60) ns 占用 CUDA 资源
        ResourceTiming(
            resource="sfu", service_time=50 * ns, offset=30 * ns,
        ),  # 在 [30, 80) ns 占用 SFU 资源
    ),
)
```

- `latency`：阶段结果何时可用，影响后继的依赖等待。
- `service_time`：某个建模资源被占用多久，影响其他阶段的资源竞争。
- `offset`：资源占用相对阶段起点的偏移。

这个例子中，两种资源在 `[30, 60)` ns 重叠使用。CUDA 在 60 ns 后释放，SFU 在 80 ns 后释放；在其他约束允许时，独立工作可以使用已释放的资源。依赖本阶段完整结果的后继仍需等到 100 ns。

**阶段完成时间是 100 ns，不是 `60 + 50 = 110 ns`，更不是 `100 + 60 + 50 = 210 ns`。** 80–100 ns 表示本例给定的“资源服务已结束、结果尚未就绪”的等待；这个间隔同样是输入假设，不是 TileSight 自动推导出的额外成本。

一个 Phase 可以使用多个不同资源。资源列表的排列顺序没有时间含义，内部先后或重叠由 `offset` 明确给出；默认 `offset=0` 表示从阶段起点开始，不表示未知。调度器移动整个 Phase，不会重新搜索它内部各资源区间的排列。

该版本不允许一个 Phase 多次声明同一资源；需要多段占用或希望内部操作也参与调度时，应拆成多个节点并补依赖。前端 `Timing` 还要求各资源区间不超出阶段完成时间。

### 5.2 Work 不会自动变成完整 Timing

`Work(kind, flops, bytes, attrs)` 是工作描述，不是现成成本。计算工作可以经官方 `bind_work_throughputs(op, arch, fallback_policy)` 获得各工作项的吞吐绑定和服务时间；这个函数消费具有 `work_items` 的语义操作对象，并非任意一个 `Work` 对象都可直接传入。

访存还需要相应的访问、缓存和层级流量分析。随后将得到的成本按明确的组合规则形成 Timing。将 dtype、工作数量或字节数填入 Work，不等于已解决成本、依赖或 grid 分析。

## 6. 连贯示例：分块矩阵乘的前端声明与完整分析

示例为 `M=N=K=8192`、`BM=BN=128`、`BK=64`，每个输出 tile 的 K 循环为 128 轮，逻辑 grid 为 `64×64`。每轮 FP16 A/B 各加载 16 KiB，矩阵乘工作量为 `2×128×128×64` FLOPs。

下面代码演示接口连通性。**所有纳秒值均为人为构造的教学输入，不是官方定标、当前项目预测或实测值；256 threads 和 1 CTA/SM 也只是本例的显式配置。** 示例把累加结果视为当前轮完成后供下一轮使用，并把写回简化为单阶段，不用于复刻真实异步 WGMMA 的细节或验证 GEMM 精度。

在可导入该固定版本 TileSight 的 Python 环境中执行，不运行 GPU kernel：

```python
# runnable-interface-demo
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.tilesight_new_api.frontend import Kernel
from tilesight.tilesight_new_api.ir import Work, Timing, ResourceTiming
from tilesight.tilesight_new_api.regions import (
    PhaseRegion, SequenceRegion, LoopRegion, PeriodicAxisIR,
)
from tilesight.tilesight_new_api.native_executor import (
    NativeModelOptions, model_native,
)
from tilesight.fused_op_pipeline_wave.periodic_schedule import (
    schedule_periodic_dag,
)

ns = 1e-9
kernel = Kernel("matmul_interface_demo")
launch = kernel.launch(
    name="main",
    work_grid=(64, 64, 1),
    physical_grid=(64, 64, 1),
    threads=256,
    cluster=(1, 1, 1),
    residency=1,
    scheduler="static",
)

# 前端周期循环：登记角色、工作和约束。
pipeline = launch.periodic("k_pipeline", iterations=128, stages=3)
a_buffer = pipeline.buffer(
    "A_shared", scope="smem", shape=(128, 64), dtype="float16", slots=3,
)
b_buffer = pipeline.buffer(
    "B_shared", scope="smem", shape=(64, 128), dtype="float16", slots=3,
)
producer = pipeline.actor("producer", execution_scope="warpgroup")
consumer = pipeline.actor("consumer", execution_scope="warpgroup")
load_a = producer.phase(
    "load_a", work=Work.copy(bytes=16384),
    timing=Timing(100 * ns, (ResourceTiming("tma", 60 * ns),)),
    writes=(a_buffer,),
)
load_b = producer.phase(
    "load_b", work=Work.copy(bytes=16384),
    timing=Timing(100 * ns, (ResourceTiming("tma", 60 * ns),)),
    writes=(b_buffer,),
)
mma = consumer.phase(
    "mma", work=Work.mma(flops=2 * 128 * 128 * 64),
    timing=Timing(180 * ns, (ResourceTiming("tensor", 180 * ns),)),
    reads=(a_buffer, b_buffer),
)
producer.sequence(load_a.at(0), load_b.at(0), order="issue")
pipeline.after(load_a.done, mma.start)
pipeline.after(load_b.done, mma.start)
accumulator = pipeline.state("accumulator")
pipeline.carry(accumulator, source=mma.done, target=mma.start, distance=1)
pipeline.pipeline_buffer(a_buffer, acquire=load_a.start, release=mma.done)
pipeline.pipeline_buffer(b_buffer, acquire=load_b.start, release=mma.done)

# 官方前端要求 Phase 归属某个周期声明。
# 此处只是登记一次性工作的归属；它们在 region 中位于循环外。
boundary = launch.periodic("boundary_work", iterations=1, stages=1)
boundary_actor = boundary.actor("boundary")
initialize = boundary_actor.phase(
    "initialize", work=Work("fill", attrs={"elements": 128 * 128}),
    timing=Timing(20 * ns, (ResourceTiming("cuda", 20 * ns),)),
)
store = boundary_actor.phase(
    "store", work=Work.copy(bytes=128 * 128 * 2),
    timing=Timing(80 * ns, (ResourceTiming("memory", 80 * ns),)),
)

# 执行结构引用同一批前端 Phase，不重新计算工作和 Timing。
root = LoopRegion(
    name="k_loop", trip_count=128,
    prologue=PhaseRegion(initialize),
    body=SequenceRegion(
        "iteration_body",
        (PhaseRegion(load_a), PhaseRegion(load_b), PhaseRegion(mma)),
    ),
    epilogue=PhaseRegion(store),
    periodic_axis=PeriodicAxisIR(
        loop_name="k_pipeline", ii_mode="periodic_best", summary_policy="macro",
        boundary_anchor_phase="load_a",
    ),
)
ir = kernel.build()

# 可选：直接查看官方前端生成的 DAG 和独立流水分析结果。
dag = ir.lower_periodic("k_pipeline")
envelope = schedule_periodic_dag(dag)
print("II (s):", envelope.best.ii)
print("phase starts (s):", dict(envelope.best.phase_starts))
print("search complete:", envelope.search_complete)

# 完整区域/grid 分析不要求事先调用上面两行；loop_name 会触发转换。
result = model_native(
    kernel=ir, region=root,
    arch=H200_SXM().set_to_microbench(),
    options=NativeModelOptions(
        launch_name="main", cost_source="phase_timing",
        ii_mode="periodic_best", boundary_policy="finite_witness",
        multi_cta_policy="resource_bound", liveness_policy="report_static",
        kernel_launch_s=0.0, host_dispatch_s=0.0,
    ),
)
print("per work unit (s):", result.per_work_unit_s)
print("kernel body (s):", result.kernel_body_s)
```

这里 `root.periodic_axis.loop_name="k_pipeline"` 按名称引用 `pipeline = launch.periodic("k_pipeline", ...)` 登记的循环声明；`root` 自己的名称 `"k_loop"` 不用于匹配。`root.body` 则引用该 pipeline 中登记的同一批 Phase。

代码中的 `boundary_work` 只是前端声明容器，不表示实际还要额外执行一次循环。一次性工作的执行位置由 root 的 prologue/epilogue 指定。`stages=3` 也不能代替 buffer 的 acquire/release 关系。

2026-09-16 已在上述固定版本上验证：本文 9 个 Python 代码块均通过语法解析；完整示例实际执行通过，生成 3 个 DAG 节点、2 个 token buffer 和累加器跨轮依赖；将 `loop_name` 方式替换为同图的显式 `dag` 方式后，工作单元时间和 kernel-body 时间一致。验证不运行 GPU，不构成性能准确率测试。

原始源程序的对应关系为：循环前 clear → initialize；循环内 copy A/B、矩阵乘 → body；循环后写回 → store。对其他程序必须按实际循环和作用域提取，不能按算子名字套用这个边界。

## 7. Region 中周期分析的细节：官方前端如何组装 PeriodicDAG

本节展开第 2、3 节的 `LoopRegion.periodic_axis`：**执行器分析到这个循环区域时，如何取得其周期 DAG。** 这不是另外启动一套与 Region 无关的完整 kernel 分析。

先解释示例里一直使用的 `pipeline`：

```python
pipeline = launch.periodic(
    "k_pipeline", iterations=128, stages=3,
)
```

`pipeline` 是我们给返回对象起的 Python 变量名，实际类型为官方前端的 `PeriodicLoop`。它是**循环内工作和约束的声明容器**，不是已经排好的流水，也不是另一种 Region。官方也提供 `launch.pipeline(...)`，它是 `launch.periodic(...)` 的别名。

在这个容器中，通过 `actor` 登记角色和 Phase，通过 `after`、`carry`、`pipeline_buffer` 等登记同轮依赖、跨轮状态与缓冲关系。`kernel.build()` 将这些声明保存为 `LaunchIR` 下的 `PeriodicLoopIR`。

`iterations=128` 是前端循环次数声明，`stages=3` 是流水级数配置；它们本身不足以构造流水，仍需登记具体的 Phase、依赖及缓冲获取／释放关系。Region 一侧用 `trip_count=128` 描述本次有限执行次数，使用时应与对应前端声明保持一致。

**因此，这里讲的是 Region 的周期分析细节，但 pipeline 对象不直接存放在 Region 内，而是由 Region 引用：**

```text
kernel 中的声明                          本次分析的 Region
LaunchIR                                LoopRegion("k_loop", trip_count=128)
└─ PeriodicLoopIR("k_pipeline")          ├─ prologue / body / epilogue
   ├─ actors → phases                   └─ periodic_axis
   ├─ dependencies / carries               └─ loop_name="k_pipeline"
   └─ buffers / 资源顺序等                            │
                ↑────────────────── 按名称查找 ──────┘
                │
       官方 lower_periodic 转换
                ↓
           PeriodicDAG
                ↓
       调度器分析 II 和阶段起点
                ↓
       Region 执行器组合有限循环及前后操作的时间
```

具体引用写法为：

```python
region = LoopRegion(
    name="k_loop",
    trip_count=128,
    body=iteration_region,          # 引用同一批前端 Phase
    prologue=initialize_region,
    epilogue=store_region,
    periodic_axis=PeriodicAxisIR(loop_name="k_pipeline"),
)
ir = kernel.build()
result = model_native(kernel=ir, region=region, arch=arch)
```

`model_native` 在求值该周期区域时，会按 `loop_name` 取得对应声明并进行转换；不要求用户事先手动排一次流水。若已经提供 `PeriodicAxisIR(dag=dag)`，则跳过下面的前端转换，直接消费这张图。

前端 `pipeline.after(...)` 创建事件依赖并写入 `PeriodicLoopIR.dependencies`；`pipeline.carry(...)` 写入 carries。这些是边及其他声明，不是第二张独立设计的完整 DAG。

`KernelIR.lower_periodic(loop_name, oracle=None)` 组合：

1. 前端 Phase 的显式 Timing，或由 oracle 解析的 Timing；
2. actor 明确声明的周期顺序；
3. 显式事件依赖、跨轮 carry；
4. 支持范围内由 buffer 读写推导的数据流；
5. buffer 槽位生命周期；
6. 明确固定的资源顺序。

生成的对象为：

```python
PeriodicDAG(
    phases=...,
    dependencies=...,
    token_buffers=...,
    fixed_resource_orders=...,
)
```

**先生成图，再分析流水。PeriodicDAG 是输入，不是已经排好流水的结果。** 硬件能力通过成本绑定等步骤进入资源时间；不能只给 actor 名称就期望它自动推导全部硬件行为。

### 7.1 直接构图可以绕过官方建模前端

等价层次的接口示意如下，变量中的时间仍需提前提供：

```python
from tilesight.fused_op_pipeline_wave.periodic_schedule import (
    Phase as SchedulePhase, Dependency as ScheduleDependency,
    ResourceUse, TokenBuffer, PeriodicDAG, schedule_periodic_dag,
)

load = SchedulePhase("load", load_latency, (ResourceUse("tma", load_service),))
mma = SchedulePhase("mma", mma_latency, (ResourceUse("tensor", mma_service),))
dag = PeriodicDAG(
    phases=(load, mma),
    dependencies=(ScheduleDependency("load", "mma"),),
    token_buffers=(TokenBuffer("tiles", "load", "mma", capacity=3),),
)
envelope = schedule_periodic_dag(dag)
```

这里只展示加载与计算，不是前面完整累加循环的替代图。直接 DAG 的 Phase 不必绑定 actor；相关程序约束需由构图方明确提供。名称在当前 DAG 内唯一，不能仅凭字符串引用另一个独立 DAG 的阶段。

### 7.2 单独分析 DAG 与完整 kernel 分析

`schedule_periodic_dag(dag)` 不需要 KernelIR、Region 或 grid。返回 `ScheduleEnvelope`，包括 best/worst 的 II、阶段起点、资源顺序、下界、搜索完整性等。best/worst 是所搜索合法资源顺序的结果，不是任意等待下的绝对最快/最慢实际执行时间；只有搜索完整等条件成立时才能作更强的全局表述。

该调用不包含循环次数和整个 grid，因此不直接给出完整 kernel latency。若想把图交给区域执行器，可使用：

```python
PeriodicAxisIR(dag=dag, ii_mode="periodic_best")
```

传入的是输入图 `dag`，不是分析结果 `envelope`。`dag` 和 `loop_name` 二选一；不要为了构造 axis 而先重复跑一遍调度。

## 8. 跨轮依赖、状态与缓冲区复用

### 8.1 Dependency 的迭代距离

**`ScheduleDependency` 属于后端 `PeriodicDAG.dependencies`，不是直接写在 `LoopRegion` 上的字段。** LoopRegion 通过 `periodic_axis` 关联 DAG。下面把声明位置一起写出；`mma` 是已经包含 Timing 信息的后端 Phase：

```python
dag = PeriodicDAG(
    phases=(mma,),
    dependencies=(
        ScheduleDependency(
            source="mma", target="mma",
            iteration_distance=1, min_delay=None,
        ),
    ),
)
region = LoopRegion(
    name="k_loop",
    trip_count=128,
    body=iteration_region,  # 引用与此 DAG 匹配的前端 Phase
    periodic_axis=PeriodicAxisIR(dag=dag),
)
```

这条边表示 `start(mma, i+1) >= start(mma, i) + latency(mma)`。`min_delay=None` 使用 source 的完整 latency；`min_delay=0` 表示开始到开始约束，不要求等 source 完成。

如果走官方前端，则写 `pipeline.after(a.done, b.start, distance=1)`，或使用有状态名称的 `StateCarry`；声明保存在 `PeriodicLoopIR` 中，经官方转换成为后端 DAG 的依赖。此时 Region 用 `PeriodicAxisIR(loop_name="k_pipeline")` 引用前端循环，不必手动再填写一份 `ScheduleDependency`。普通同轮数据流 distance 为 0。后端也支持有符号距离来表达展开窗口中的操作位置，但必须满足其因果性限制，不能任意写负距离绕过约束。

周期模板可能包含带距离的回边或自环；边连接不同迭代的实例，不等于单轮存在非法循环依赖。

### 8.2 TokenBuffer 不是 ResourceUse

```python
TokenBuffer(name="tiles", acquire="load", release="mma", capacity=3)
```

默认从 load 开始占槽，到 mma 完成释放，约束后面的 `load(i+3)` 不能提前复用 `mma(i)` 尚未释放的槽位。

| 信息 | 对应对象 | 单位/含义 |
|---|---|---|
| 存储容量 | 前端 Buffer 的 shape/dtype/slots 等 | 存储字节需求，参与生命周期/容量分析 |
| 同时在途的份数 | PipelineBuffer / TokenBuffer | 槽位或 token 数，不含每槽字节数 |
| 访问硬件资源的服务 | ResourceTiming / ResourceUse | 秒及相对起点偏移 |

数据留在 shared memory 中等待使用，不表示一直占用 SMEM 带宽。槽位、容量和带宽不能相互替代。

## 9. grid、硬件能力与时间口径

`LaunchIR` 提供 `work_grid`、`physical_grid`、`threads`、`cluster`、`residency`、`scheduler`、`swizzle` 等声明，均不是 Phase 自身的字段。

- work_grid：逻辑工作单元数量。
- physical_grid：启动的物理 CTA 数量。
- 普通 kernel 二者往往相同；persistent 程序二者可以不同，但仍需受支持的工作分派/空间组合逻辑，不能只修改两个数就声称完成建模。
- residency：模型采用的每 SM 驻留 CTA 数。这个固定版本的 native 路径在未明确给出时回退到 1，不能声称它仅凭 threads 自动准确计算最终寄存器限制。
- 多 CTA 使用 `resource_bound` 是指定的组合近似，不等价于已经构建每个 CTA 的完整联合阶段图。

时间应分别报告阶段完成时间、资源服务时间、II、有限 CTA/工作单元完成时间、kernel-body latency、launch/host 开销。不能把 II 当作 kernel latency，也不能把所有阶段时间简单相加后当作流水结果。

## 10. 能力边界：技术报告必须披露的内容

| 边界 | 当前固定版本的行为 | 对前端的要求 |
|---|---|---|
| 条件分支 | 没有通用 IfRegion/SwitchRegion，不自动选择互斥分支 | 先解析配置、循环坐标、线程参与范围及必要输入；不能把两个互斥分支同时当作必做工作 |
| 嵌套周期循环 | 周期宏的 body 不能包含未展开内层 LoopRegion；执行器不自动将它展平成联合 DAG | 外层串行、内层周期可以表达；跨层重叠需显式转换或披露分层组合限制 |
| 周期 inline 配置 | 虽有字段，该 native 实现明确拒绝周期 `summary_policy="inline"` | 字段存在不代表执行方式已经实现 |
| 跨独立区域重叠 | Sequence 顺序组合区域，不自动合并多个独立周期 DAG | 需要联合重叠的工作应进入同一受支持 DAG；保留真实依赖 |
| Phase 内部调度 | 按给定 offset 使用固定资源区间，不重新安排内部操作 | 需要内部调度自由度时拆分节点 |
| actor 与资源 | actor 名称不自动产生资源或物理发射成本 | 明确 Timing 与程序顺序，不按名称猜测 |
| 前端 buffer 自动数据流 | 当前转换要求一个 buffer 至多一个语义 producer，并把自动 producer→consumer 流视为同轮 | 多 writer、版本化/跨轮别名不能期待自动推导；需正确规范化或直接构图 |
| 调度搜索结果 | 搜索可能不完整，worst 不包含任意插入的空闲 | 保留 search_complete、搜索范围和假设 |
| 高层来源的精度 | source 不直接给出最终寄存器、spill、机器指令安排 | 区分源层事实和执行假设，不宣称等价于编译后实测 |

在我们的输入策略中，影响工作量或结构的数据依赖若无法仅凭元数据求值，就要求显式输入；提供后仍无法求值则报告错误。这是 adapter 的边界策略，不是 TileSight 原生的分支分析能力。

也不能把“任意两个 Region 都不能重叠”当成结论：同一个周期 body 下的多个 PhaseRegion 可以进入同一张 DAG 联合分析。真正需要说明的是 **Region 执行器不会自动跨独立区域建立全局联合流水图**。

## 11. 与我们当前实际调用链的关系

上面的完整例子演示官方“前端声明 → DAG → 原生区域执行器”的使用方式，不代表 adapter 当前所有路径都走这套 builder。

当前源码流水路径主要是：

```text
Python builder → 高层 TIR
    → 我们的语义事实、值、工作量和依赖解析
    → 源层有限执行实例，以及可证明的周期结构
    → 官方成本能力/缓存分析 → Timing
    → 直接组装官方 PeriodicDAG，调用官方流水/有限执行辅助能力
    → CTA 完成时间及资源服务量汇总
    → model_native 的 grid/多 CTA 组合
    → 结果与报告
```

对应模块为 [source_pipeline.py（实现仓库中的 `src/tilesight_for_tilelang/model/backends/tilesight/source_pipeline.py`）、source_pipeline_costs.py（实现仓库中的 `src/tilesight_for_tilelang/model/backends/tilesight/source_pipeline_costs.py`） 和 source_pipeline_grid.py（实现仓库中的 `src/tilesight_for_tilelang/model/backends/tilesight/source_pipeline_grid.py`）。

其中 CTA 汇总 Phase 只是 grid 接口转换对象，不是新增的语义 tile 操作。原始阶段图、成本与执行关系应继续保留，不能用汇总 Phase 冒充原流水图。报告中也不能声称已完成官方任意嵌套 Region 的通用自动接入。

## 12. 给前端设计的结论

我们的核心输出不是一个 kernel 类别，而是：

1. 操作、数据对象、有效范围和工作量；
2. 阶段间的数据、同步、状态和存储复用关系；
3. 循环次数、边界、参与范围与 launch 环境；
4. 与这些事实一致的官方成本输入和 Timing；
5. 应放在同一 DAG 中联合分析的范围。

抽取这些事实后，可以选择官方前端转换，或直接构造官方 DAG。后者绕过的是 TileSight 的建模 builder，不是绕过我们的语义解析；也意味着约束展开的正确性由我们负责。

一句话总结：**Phase 是工作和成本的单位，DAG 是流水约束的单位，Region 是程序组合的单位，Launch 是空间执行环境。四者各司其职，不可互相替代。**

## 13. 官方源码索引

以下均指上述固定版本，核对或升级接口时以实现为准：

- `tilesight/tilesight_new_api/ir.py`：Work、Timing、前端 Phase、Dependency、StateCarry、ActorIR、PeriodicLoopIR、LaunchIR、KernelIR。
- `tilesight/tilesight_new_api/frontend.py`：Kernel/Launch/PeriodicLoop/Actor builder，after/carry/sequence/pipeline_buffer。
- `tilesight/tilesight_new_api/lowering.py`：lower_periodic、actor 顺序、buffer 流和 token 转换。
- `tilesight/tilesight_new_api/regions.py`：三种 Region、PeriodicAxisIR。
- `tilesight/tilesight_new_api/native_executor.py`：model_native、区域递归求值、周期 body 校验、嵌套与 inline 限制、默认 residency。
- `tilesight/fused_op_pipeline_wave/periodic_schedule.py`：后端 Phase、Dependency、TokenBuffer、PeriodicDAG、ScheduleEnvelope、schedule_periodic_dag。
- `tilesight/tilesight_new_api/ops/throughput.py`：bind_work_throughputs。
- `tilesight/tilesight_new_api/examples/native_nested_regions.py`、`fusion_gemm_bias.py`：官方嵌套区域和显式 DAG 示例。
