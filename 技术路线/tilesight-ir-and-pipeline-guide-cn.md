# TileSight 的阶段、前端接口与流水分析

本文用于项目技术介绍：先说明 TileSight 消费什么，再解释我们的前端需要生成什么。不展开 TileSight 内部的调度求解算法。

接口依据：官方 TileSight 固定版本 `48e4158459bee5df830ae4ab7dda541edaa3dc4d`。本文描述该版本，不承诺后续版本接口不变。

## 1. 从前端 Phase 到后端流水排程

### 1.1 前端 Phase：描述一项需要安排执行的工作

接入 TileSight，首先要把 kernel 中的工作整理成前端 **Phase（阶段）**。例如加载一个 tile、一次矩阵乘累加、一组逐元素计算，都可以由 Phase 表示。它是建模时划分的工作单位，不要求等于一条机器指令，也不要求与一个 TileLang 调用一一对应。

前端类型是 `tilesight.tilesight_new_api.ir.Phase`。下面将 dataclass 自动生成的构造签名展开，省略实现；`Tuple`、`Optional`、`Sequence` 为 Python 类型注解：

```python
class Phase:
    def __init__(
        self,
        name: str,
        actor: str,
        owner: str,
        work: Work,
        timing: Optional[Timing] = None,
        reads: Tuple[Buffer, ...] = (),
        writes: Tuple[Buffer, ...] = (),
    ) -> None: ...
```

| 字段 | 含义 |
|---|---|
| `name` | 阶段名称，用于在所属循环的声明中引用该阶段；不是全局 kernel 名称 |
| `actor` | 负责这项工作的执行角色名称，例如 producer、consumer；不是资源名称或线程编号 |
| `owner` | 所属前端循环的内部身份，用来检查阶段与 Buffer 等对象是否属于同一个循环；通常由构建接口填写 |
| `work` | 做什么、做多少，例如矩阵乘 FLOPs、搬运字节数及 dtype 等属性；不是耗时 |
| `timing` | 这项工作的完成延迟和资源占用。可直接提供；未提供时，后续转换需要通过成本解析入口取得，不能当作零耗时 |
| `reads` / `writes` | 读取、写入哪些前端 Buffer；供数据流及存储相关约束使用，不等于完整的依赖边集合 |

实际声明时，通常不直接填写 `actor` 和 `owner`，而是通过 `tilesight.tilesight_new_api.frontend.Actor.phase` 创建。其方法签名是：

```python
def phase(
    self,
    name: str,
    *,
    work: Work,
    timing: Optional[Timing] = None,
    reads: Sequence[Buffer] = (),
    writes: Sequence[Buffer] = (),
) -> Phase: ...

# consumer 是已经创建的 Actor；work、timing、buffer 由调用方提供。
mma = consumer.phase(
    "mma",
    work=mma_work,
    timing=mma_timing,
    reads=(a_tile, b_tile),
    writes=(accumulator,),
)
```

该方法会填写执行角色和所属循环，并登记 Phase。**依赖关系、缓冲容量、actor 顺序等在循环／pipeline 层声明，不都塞在 Phase 字段中；grid 和 CTA 驻留等执行环境也在 Phase 之外。** 这些声明共同构成排流水的条件。

### 1.2 TileSight 的职责：将阶段及约束转换为 DAG，再安排执行

就流水分析而言，TileSight 的职责是：在给定工作成本、依赖、缓冲容量和资源竞争条件下，安排这些 Phase 在不同循环迭代中的执行实例，得到阶段起点、稳态 II 和资源使用顺序。它不会自动把阶段继续拆成机器指令，也不会仅凭 Phase 名称猜出 kernel 的语义。

前端声明进入周期流水分析的关系是：

```text
前端 Phase：Work、Timing、actor、读写 Buffer
  ＋ 所属循环中的依赖、跨轮状态、缓冲容量和必要顺序
                 ↓ 解析每个阶段的 Timing，转换各项约束
PeriodicDAG
  ├─ phases：后端 Phase
  ├─ dependencies：调度依赖
  ├─ token_buffers：容量约束
  └─ fixed_resource_orders：明确固定的资源顺序
                 ↓ schedule_periodic_dag(dag)
流水排程结果：阶段起点、II、资源顺序及相关边界信息
```

官方转换入口为 `lower_periodic(kernel: KernelIR, loop: str, oracle=None) -> PeriodicDAG`，也可以调用 `KernelIR.lower_periodic(loop_name, oracle=None)`。它在构建 DAG 时生成后端 Phase。**后端 Phase 是排程输入，不是排完流水后才产生的结果；阶段的绝对起点不保存在 Phase 构造参数中。** 有限循环和完整 kernel 时间还需结合后文的 Region、循环次数和 launch 执行环境计算。

### 1.3 后端 Phase：调度器直接消费的节点

后端类型是 `tilesight.fused_op_pipeline_wave.periodic_schedule.Phase`。它与前端 Phase 同名，但不是同一个类型。本文将导入别名写成 `SchedulePhase`，其构造签名为：

```python
class SchedulePhase:  # 官方类名为 Phase，此处使用别名区分层次
    def __init__(
        self,
        name: str,
        latency: float,
        resources: Tuple[ResourceUse, ...] = (),
        iteration_offset: int = 0,
    ) -> None: ...

class ResourceUse:
    def __init__(
        self,
        resource: str,
        service_time: float,
        offset: float = 0.0,
    ) -> None: ...
```

| 字段 | 含义 |
|---|---|
| `name` | 当前 DAG 内的节点名称，依赖边据此引用它 |
| `latency` | 从阶段开始到阶段完成的延迟，单位秒；不是整个 kernel 的时间 |
| `resources` | 使用哪些建模资源、每种资源占用多久、从阶段开始后多久占用；用于排布资源竞争 |
| `iteration_offset` | 在展开的调度窗口中，该阶段对应哪个逻辑迭代，通常为 0；不是秒数，也不是循环次数或依赖边的跨轮距离 |

前后两层的核心映射为：

```text
前端 phase.name                   → 后端 phase.name
解析后的 timing.latency           → 后端 phase.latency
timing.resources: ResourceTiming   → 后端 phase.resources: ResourceUse
actor/resource 序列中登记的
  phase.at(...) 窗口声明           → 后端 phase.iteration_offset
其余执行顺序、数据和存储约束       → DAG 的依赖、容量及固定资源顺序
```

因此，后端 Phase 不再保存 `Work`、`actor`、`reads`、`writes`：工作已落实为 Timing，相关执行语义已转换为调度约束。调度器直接消费这一层；它不需要再从 Python 代码重新推导这些事实。

两层 Phase 及转换的固定版本源码见 [前端定义](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/ir.py)、[声明接口](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/frontend.py)、[DAG 转换](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/tilesight_new_api/lowering.py) 和 [后端定义](https://github.com/tile-ai/TileSight/blob/48e4158459bee5df830ae4ab7dda541edaa3dc4d/tilesight/fused_op_pipeline_wave/periodic_schedule.py)。

### 1.4 完成时间与资源占用不是相加关系

前端时间对象的关键字段为：

```python
Timing(
    latency=...,  # 从阶段开始到完成，单位秒
    resources=(
        ResourceTiming(resource="tensor", service_time=..., offset=...),
    ),
)
```

- `latency`：阶段结果何时可用，影响后继的依赖等待。
- `service_time`：某个建模资源被占用多久，影响其他阶段的资源竞争。
- `offset`：资源占用相对阶段起点的偏移。

例如阶段完成需要 100 ns，而资源只在 `[0, 30)` ns 被占用，则独立工作可以在资源释放后使用它，但依赖该阶段结果的工作仍需等到 100 ns。不是 `100 + 30 = 130 ns`。

一个 Phase 可以使用多个不同资源。资源列表的排列顺序没有时间含义，内部先后或重叠由 `offset` 明确给出；默认 `offset=0` 表示从阶段起点开始，不表示未知。调度器移动整个 Phase，不会重新搜索它内部各资源区间的排列。

该版本不允许一个 Phase 多次声明同一资源；需要多段占用或希望内部操作也参与调度时，应拆成多个节点并补依赖。前端 `Timing` 还要求各资源区间不超出阶段完成时间。

### 1.5 Work 不会自动变成完整 Timing

`Work(kind, flops, bytes, attrs)` 是工作描述，不是现成成本。计算工作可以经官方 `bind_work_throughputs(op, arch, fallback_policy)` 获得各工作项的吞吐绑定和服务时间；这个函数消费具有 `work_items` 的语义操作对象，并非任意一个 `Work` 对象都可直接传入。

访存还需要相应的访问、缓存和层级流量分析。随后将得到的成本按明确的组合规则形成 Timing。将 dtype、工作数量或字节数填入 Work，不等于已解决成本、依赖或 grid 分析。

## 2. 从整体调用入口向内展开

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

整体对象关系如下：

```text
model_native
├─ kernel: KernelIR
│  └─ LaunchIR
│     ├─ work_grid / physical_grid / threads / cluster / residency
│     └─ PeriodicLoopIR
│        ├─ ActorIR → 前端 Phase → Work、Timing、读写 buffer
│        ├─ dependencies / carries
│        └─ lifetimes / pipeline_buffers / resource_sequences
├─ region: 递归执行结构
│  ├─ PhaseRegion → 引用一个前端 Phase
│  ├─ SequenceRegion → children
│  └─ LoopRegion → body、trip_count、prologue、epilogue
│                 └─ 可选 periodic_axis → loop_name 或显式 DAG
├─ arch
└─ options / 可选 oracle
```

这些是建模对象，不是 TileLang/TVM 编译器中的原始 TIR 节点。

## 3. Region：程序如何组织与重复

当前公共 Region 类型主要有三种：

| 类型 | 关键字段 | 普通区域执行语义 |
|---|---|---|
| `PhaseRegion` | `phase`、可选 `label` | 一个工作叶子，不是区域入口标记 |
| `SequenceRegion` | `name`、`children` | 子区域按完成后再开始的顺序组合 |
| `LoopRegion` | `name`、`trip_count`、`body`、可选前后区域及周期配置 | 重复 body；未绑定周期配置时按串行重复分析 |

`SequenceRegion` 没有 `body` 或 `dependencies` 字段。`LoopRegion` 的 body、prologue 和 epilogue 都可以是其他 Region，因此可以递归嵌套。

`PhaseRegion(load_a)` 表示该叶子包含 load_a 这项工作。阶段的时间端点是 `load_a.start` 和 `load_a.done`，不由 PhaseRegion 表示。

### 3.1 为什么已有循环次数，还需要周期 DAG

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

例如 TIR 循环变量 `k`、Region 名称 `k_loop`、前端循环名 `k_pipeline` 可以不同。前端建立关联，TileSight 不按名称相似性猜测对应关系。

## 4. Actor：执行角色，不是硬件资源本身

ActorIR 保存 `name`、`phases`、`sequence`、`order`、`execution_scope`、可选 `execution_domain` 及兼容性字段 `serial_resource`。

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

## 5. 连贯示例：分块矩阵乘的前端声明与完整分析

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

代码中的 `boundary_work` 只是前端声明容器，不表示实际还要额外执行一次循环。一次性工作的执行位置由 root 的 prologue/epilogue 指定。`stages=3` 也不能代替 buffer 的 acquire/release 关系。

2026-09-16 已在上述固定版本上验证：本文 9 个 Python 代码块均通过语法解析；完整示例实际执行通过，生成 3 个 DAG 节点、2 个 token buffer 和累加器跨轮依赖；将 `loop_name` 方式替换为同图的显式 `dag` 方式后，工作单元时间和 kernel-body 时间一致。验证不运行 GPU，不构成性能准确率测试。

原始源程序的对应关系为：循环前 clear → initialize；循环内 copy A/B、矩阵乘 → body；循环后写回 → store。对其他程序必须按实际循环和作用域提取，不能按算子名字套用这个边界。

## 6. 官方前端如何组装 PeriodicDAG

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

### 6.1 直接构图可以绕过官方建模前端

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

### 6.2 单独分析 DAG 与完整 kernel 分析

`schedule_periodic_dag(dag)` 不需要 KernelIR、Region 或 grid。返回 `ScheduleEnvelope`，包括 best/worst 的 II、阶段起点、资源顺序、下界、搜索完整性等。best/worst 是所搜索合法资源顺序的结果，不是任意等待下的绝对最快/最慢实际执行时间；只有搜索完整等条件成立时才能作更强的全局表述。

该调用不包含循环次数和整个 grid，因此不直接给出完整 kernel latency。若想把图交给区域执行器，可使用：

```python
PeriodicAxisIR(dag=dag, ii_mode="periodic_best")
```

传入的是输入图 `dag`，不是分析结果 `envelope`。`dag` 和 `loop_name` 二选一；不要为了构造 axis 而先重复跑一遍调度。

## 7. 跨轮依赖、状态与缓冲区复用

### 7.1 Dependency 的迭代距离

流水层的边为：

```python
ScheduleDependency(
    source="mma", target="mma",
    iteration_distance=1, min_delay=None,
)
```

表示 `start(mma, i+1) >= start(mma, i) + latency(mma)`。`min_delay=None` 使用 source 的完整 latency；`min_delay=0` 表示开始到开始约束，不要求等 source 完成。

前端事件依赖还可直接写 `pipeline.after(a.done, b.start, distance=1)`，或使用有状态名称的 `StateCarry`。普通同轮数据流 distance 为 0。后端也支持有符号距离来表达展开窗口中的操作位置，但必须满足其因果性限制，不能任意写负距离绕过约束。

周期模板可能包含带距离的回边或自环；边连接不同迭代的实例，不等于单轮存在非法循环依赖。

### 7.2 TokenBuffer 不是 ResourceUse

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

## 8. grid、硬件能力与时间口径

`LaunchIR` 提供 `work_grid`、`physical_grid`、`threads`、`cluster`、`residency`、`scheduler`、`swizzle` 等声明，均不是 Phase 自身的字段。

- work_grid：逻辑工作单元数量。
- physical_grid：启动的物理 CTA 数量。
- 普通 kernel 二者往往相同；persistent 程序二者可以不同，但仍需受支持的工作分派/空间组合逻辑，不能只修改两个数就声称完成建模。
- residency：模型采用的每 SM 驻留 CTA 数。这个固定版本的 native 路径在未明确给出时回退到 1，不能声称它仅凭 threads 自动准确计算最终寄存器限制。
- 多 CTA 使用 `resource_bound` 是指定的组合近似，不等价于已经构建每个 CTA 的完整联合阶段图。

时间应分别报告阶段完成时间、资源服务时间、II、有限 CTA/工作单元完成时间、kernel-body latency、launch/host 开销。不能把 II 当作 kernel latency，也不能把所有阶段时间简单相加后当作流水结果。

## 9. 能力边界：技术报告必须披露的内容

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

## 10. 与我们当前实际调用链的关系

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

## 11. 给前端设计的结论

我们的核心输出不是一个 kernel 类别，而是：

1. 操作、数据对象、有效范围和工作量；
2. 阶段间的数据、同步、状态和存储复用关系；
3. 循环次数、边界、参与范围与 launch 环境；
4. 与这些事实一致的官方成本输入和 Timing；
5. 应放在同一 DAG 中联合分析的范围。

抽取这些事实后，可以选择官方前端转换，或直接构造官方 DAG。后者绕过的是 TileSight 的建模 builder，不是绕过我们的语义解析；也意味着约束展开的正确性由我们负责。

一句话总结：**Phase 是工作和成本的单位，DAG 是流水约束的单位，Region 是程序组合的单位，Launch 是空间执行环境。四者各司其职，不可互相替代。**

## 12. 官方源码索引

以下均指上述固定版本，核对或升级接口时以实现为准：

- `tilesight/tilesight_new_api/ir.py`：Work、Timing、前端 Phase、Dependency、StateCarry、ActorIR、PeriodicLoopIR、LaunchIR、KernelIR。
- `tilesight/tilesight_new_api/frontend.py`：Kernel/Launch/PeriodicLoop/Actor builder，after/carry/sequence/pipeline_buffer。
- `tilesight/tilesight_new_api/lowering.py`：lower_periodic、actor 顺序、buffer 流和 token 转换。
- `tilesight/tilesight_new_api/regions.py`：三种 Region、PeriodicAxisIR。
- `tilesight/tilesight_new_api/native_executor.py`：model_native、区域递归求值、周期 body 校验、嵌套与 inline 限制、默认 residency。
- `tilesight/fused_op_pipeline_wave/periodic_schedule.py`：后端 Phase、Dependency、TokenBuffer、PeriodicDAG、ScheduleEnvelope、schedule_periodic_dag。
- `tilesight/tilesight_new_api/ops/throughput.py`：bind_work_throughputs。
- `tilesight/tilesight_new_api/examples/native_nested_regions.py`、`fusion_gemm_bias.py`：官方嵌套区域和显式 DAG 示例。
