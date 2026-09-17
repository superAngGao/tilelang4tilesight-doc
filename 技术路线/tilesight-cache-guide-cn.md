# TileSight 缓存分析：接口、数据流与能力边界

本文用于项目技术报告及接口讨论，与 [TileSight 接口与流水分析](tilesight-ir-and-pipeline-guide-cn.md) 配套。按“分析入口 → 输入对象 → 命中率计算 → Phase 时间对接”的顺序介绍，不把缓存分析描述成任意程序的精确访存模拟。

依据：官方 TileSight 固定版本 `48e4158459bee5df830ae4ab7dda541edaa3dc4d`；本项目接线依据 `b15b4ac`。后续版本可能变化。本文不是 GPU 实测精度报告，也不新增或修改模型实现。

## 1. 先区分两项功能

| 功能 | 直接输入 | 直接输出 |
|---|---|---|
| 缓存分析 | `CacheProblem`：数据访问、复用身份、工作网格、遍历方式、缓存配置 | 命中率、分层流量、复用距离统计 |
| 流水分析 | Phase 的 Timing、依赖、缓冲容量和必要顺序等 | 阶段起点、II、可行资源顺序等 |

两者有独立入口。缓存分析不直接消费 `LoopRegion`、`SequenceRegion`、Actor 或 PeriodicDAG，也不直接输出 Phase Timing。

当前组合方向是：

```text
源层访问事实 + 工作块遍历假设 + 缓存配置
                    ↓
              CacheProblem
                    ↓ model_cache
       各类访问的命中率及分层流量
                    ↓ 映射到 Phase，统一计量范围
            官方带宽成本 → Timing
                    ↓
          与计算成本一起进入流水分析
```

不是“先排出精确流水，再按每次请求的时刻模拟缓存，再反复修正流水”。

## 2. TileSight 直接消费的入口

```python
from tilesight.tilesight_new_api.cache import model_cache

# problem 是已构造的 CacheProblem。
result = model_cache(problem)
```

接口签名为 `model_cache(problem: CacheProblem) -> CacheResult`。内部可拆成两步：

```python
from tilesight.tilesight_new_api.cache import (
    build_reuse_histogram,
    evaluate_histogram,
)

histogram = build_reuse_histogram(problem)
result = evaluate_histogram(problem, histogram)
```

第一步建立复用距离统计；第二步根据配置评估命中概率、汇总流量。中间结果带有后端及计量口径，不能任意混用。

输出结构：

```text
CacheResult
├─ problem_digest  输入标识
├─ backend         使用的缓存模型后端
├─ histogram       复用距离、采样及近似信息
├─ per_access      按访问名称分组的结果
├─ aggregate       汇总命中率
├─ traffic         汇总流量
└─ diagnostics     假设与限制
```

## 3. 输入对象：CacheProblem

以下展示真实字段；省略号是待填对象，不是可直接运行的示例：

```python
CacheProblem(
    grid=...,                       # TileGrid：工作块坐标空间
    accesses=(...),                 # CacheAccessIR：每个工作块的访问描述
    traversal=...,                  # 工作块遍历方式
    l2=...,                         # CacheLevelConfig
    l1_5=None,                      # 可选 L1.5 配置
    l1_5_group_size=0,
    inner_iterations=1,
    reduction=ReductionConfig(),
    sampling=SamplingConfig(),
    backend="tile_reuse_distance",
)
```

它围绕“规则工作网格 + 每个工作块的访问模式 + 可选内层循环”设计，不是递归程序解释器。

| 来源结构 | 与接口的关系 |
|---|---|
| GEMM 的规则 tile 循环 | 可以自然映射工作网格、输入 tile 复用及内层次数 |
| 每个 CTA 执行相同的 elementwise 读写，无内层循环 | 可以设置 `inner_iterations=1` |
| 任意 Sequence，包含多个不同循环、条件和访问范围 | 不能仅靠一个次数表达，需转换或明确分段近似 |
| 多个 kernel 的 launch 序列 | 没有直接传入跨 kernel 执行历史及缓存初始状态的通用接口 |

`inner_iterations=1` 表示没有要近似处理的多轮内层访问，不意味着自动支持任意顺序程序。分别调用 `model_cache` 分析两个 Region，也不会自动延续前一个分析的缓存状态。

## 4. 一类访问的两层描述

```text
CacheAccessIR：怎么访问
    └─ TensorTileRegion：访问什么数据
```

### 4.1 TensorTileRegion：数据区域，不是执行 Region

| 字段 | 含义 |
|---|---|
| `value` | 数据对象的逻辑身份 |
| `index_map` | 工作块坐标到数据 tile 身份的映射 |
| `tile_shape` | 访问的数据块形状 |
| `element_bytes` | 元素字节数 |
| `payload_bytes` | 有效数据量，默认形状元素数乘元素字节数 |
| `allocation_bytes` | 模型中的缓存占用量，默认等于有效数据量 |

同一 `value` 的访问必须满足接口要求的身份映射与占用量一致性；不能仅凭名字相同就合并不相容的数据区域。

`Projection(axes=(0,))` 表示只保留工作坐标的第 0 维。例如网格坐标 `(m,n)` 中，A 的身份只依赖 m，意味着不同 n 的工作块能够复用 A。B 则可用 `Projection((1,))`。

这说明某一代表 K 轮的复用身份；不同 K 轮的数据由内层循环近似处理，不能误读为所有 K 轮都访问同一个 A/B tile。

`AffineIndexMap` 支持整数仿射映射以及可选取模，但它仍然是 **tile 身份映射**，不是每个元素、每条 cacheline 或每个线程的物理地址表达式。部分重叠的数据区域也不能仅靠不同 tile key 自动识别复用。

### 4.2 CacheAccessIR：访问方式及计量

| 字段 | 含义 |
|---|---|
| `name` | 访问名称，也是查找输出及映射回 Phase 的键 |
| `region` | 对应的数据区域 |
| `mode` | `read`、`write` 或 `atomic` |
| `repetitions` | 顺序重复访问，会影响缓存访问历史 |
| `frequency_weight` | 统计权重，不等价于额外执行访问事件 |
| `transaction_bytes` | 建模的事务字节量，默认有效数据量 |
| `write_policy` | 写分配、写传播等策略；写与原子访问需要提供 |
| `include_in_hit_rate` | 是否计入汇总命中率，不能用它代替是否发生访存 |
| `placement` | 普通工作块访问，或兼容用的 wave 边界写入位置 |
| `enabled` | 是否启用该访问 |

数据对象身份、有效字节、缓存占用字节、事务字节各有用途。不得混用；`mode="atomic"` 也不代表该接口已经模拟原子竞争及串行化延迟。

## 5. 工作块大小、grid 与 CTA swizzle

内存分析需要知道数据块大小，但不是直接把 CTA 线程数当成数据量。

| 信息 | 例子 | 作用 |
|---|---|---|
| 线程块大小 | 256 threads | 影响执行及驻留条件，不直接等于缓存访问量 |
| 数据 tile 形状 | BM=128、BN=128、BK=64 | 确定单轮 A/B 访问大小 |
| 元素大小 | FP16，2 字节 | 换算字节数 |
| 工作网格 | 64×64 | 确定有哪些工作块 |
| 内层次数 | K/BK=128 | 确定多轮访问规模及缓存压力 |

上述 A 和 B 单轮各读取 `128×64×2=16384` 字节。

`TileGrid(shape=(64,64), axes=("m","n"))` 只表示工作块坐标；访问顺序由 `traversal` 单独提供。

- `RowMajorTraversal(wave_size, sm_count)`：行优先工作坐标遍历及 wave 配置。
- `PanelTraversal(row_panel, sm_count, column_panel=None, raster_axis="legacy")`：二维 panel 遍历；`raster_axis` 可为 `legacy`、`along_m`、`along_n`。

CTA swizzle 应转换成对应的工作块遍历，而不是直接填一个“提高命中率”的开关。

当前实现还会在 wave 内使用固定随机种子重排工作块；模型中的 wave 和 SM 分配是执行近似，不是 CUPTI/NCU 记录的实际调度。

## 6. 架构配置与内层循环处理

### 6.1 缓存配置没有直接接收 arch 对象

```python
CacheLevelConfig(
    name="l2",
    capacity_bytes=...,
    unit_bytes=...,
    associativity=8,
    cacheline_bytes=128,
    distance_unit="tile_allocation",
)
```

容量等信息应从目标架构配置取得，但相联度、复用单位等可能是模型参数或假设，必须区分来源。

官方 `gemm_cache_problem(...)` 便捷函数接收 `l2_capacity_bytes`、`sm_count` 等标量再组装对象；本身不接收 Architecture。当前其中还存在固定的相联度及 cacheline 参数，不能把它们都称为硬件测量结果。

`capacity_units = capacity_bytes / unit_bytes`，复用距离采用相同单位。`tile_allocation` 的 `unit_bytes` 不必等于 cacheline 大小。旧兼容后端 `legacy_volume` 则要求 `legacy_cacheline_volume` 口径，不得混用。

L1.5 是该模型使用的层级抽象，不能未经口径核对就把它当作 NCU 的 L1/TEX。SMEM 是显式存储空间，另计访问成本，不是这里的 L1.5。

### 6.2 内层循环并非任意嵌套循环的逐事件模拟

`ReductionConfig` 支持：

- `anonymous_inner`：用其他轮次造成的匿名不同数据占用来近似缓存压力。
- `stable_shadow_cohort`：选择代表 K 轮，用稳定的前后数据集合近似其他轮次；含 wave 内进度近似。
- 可选 `ProgressJitterConfig`：扰动查询用的距离，近似 CTA 进度偏差；不更新缓存状态为一条真实时间轨迹。

`SamplingConfig(seed=0, sample_budget=None)` 不启用请求预算抽样。但 wave 内随机化和内层循环近似仍可能存在，因此“取消抽样”不等于“精确硬件模拟”。

## 7. 命中率如何计算

默认 `tile_reuse_distance` 后端的计算链为：

```text
grid + traversal + 各访问身份/占用量
                    ↓
         构造模型中的访问事件
                    ↓
       计算加权的不同数据复用距离
                    ↓
          SDCM 命中概率估计
                    ↓
           汇总命中率和流量
```

例如 `A → B → C → A`，再次访问 A 的复用距离取决于介入的不同数据 B、C 的占用量。反复访问 B 不应被计成不断增加不同数据。首次访问按冷访问处理。

官方 `sdcm(D,A,B)` 中，D 是复用距离，A 是相联度，B 是同单位容量。它以 `p=A/B` 估计其他数据竞争同一缓存集合的概率，判断竞争者少于 A 的概率；小距离使用二项求和，较大距离采用正态近似，并有边界处理。

这是“访问序列/复用距离 + 概率模型”，不只是蒙特卡洛抽样，也不是逐 cacheline 的真实替换策略模拟。

### 7.1 命中率分母必须明确

| 输出字段 | 含义 |
|---|---|
| `per_access.l1_5_hit_rate` | 该类全部请求中由 L1.5 命中的比例 |
| `per_access.l2_served_rate` | 该类全部请求中最终由 L2 服务的比例 |
| `per_access.l2_hit_rate_of_l1_5_misses` | 未命中 L1.5 的请求中，命中 L2 的比例 |
| `per_access.ddr_miss_rate` | 该类全部请求中需访问 DDR 的比例 |

有请求时，L1.5 命中比例 + L2 服务比例 + DDR miss 比例等于 1。L2 条件命中率不能与另两项直接相加。

`aggregate` 对参与汇总的访问采用有效字节加权；其中 `l2_hit_rate` 是未由 L1.5 服务部分的条件命中率。它不等同于任意 NCU sector/request 指标的分母。

流量输出区分有效读写、各层请求、DDR 读写、RFO、dirty/writeback 等。字段存在不代表所有物理行为均精确模拟；例如当前 write-back 路径不会预测完整的脏数据驱逐写回，必须阅读 diagnostics。

## 8. 从缓存统计到 Phase Timing

缓存分析范围可以覆盖整个工作网格，输出按访问名分组；Phase Timing 则描述一次阶段执行。二者不是相同粒度。

前端必须保留：

```text
semantic access IDs ↔ cache access name ↔ phase ID
```

对于已证明每次访问大小及执行条件一致的类别，可以将该类累计分层流量除以对应 `request_count`，得到单次平均流量，再组成 Phase 的访存需求。一个 Phase 可以关联多类访问。

这里的“累计”必须限定为缓存模型所代表的访问统计范围，不保证已经展开全部内层轮次。例如第 9 节中 K 方向为 128 轮，但该后端输出每类访问 `request_count=4096`，对应代表访问统计，并非实际总共只执行 4096 次读取。内层次数还用于干扰近似，不能看到有 `inner_iterations` 就认定返回流量已经是整个 kernel 的全部字节。总量汇总需要依据访问的实际执行次数另行换算，避免漏乘或重复乘。

不能把全网格累计字节直接绑定到单次 Phase；也不能对不同访问量的实例无条件平均。首轮冷、后续热的差异可能被平均化，输出不是每个动态 Phase 实例的实测或精确时间。

当前正式接线的 `memory_costs.py`：

1. `_cache_access_evidence` 保存访问名与 Phase/语义来源对应，按请求数转换字节范围。
2. `_phase_memory_timing` 汇总该 Phase 的 DDR/L2/L1.5 流量，另加入源层统计的 SMEM 读写需求。
3. 调用官方 `make_op_group(..., ddr_io=..., l2_io=..., l1_5_io=..., smem_io=...)` 计算资源时间。
4. 将官方结果封装为 `Timing` 和各项 `ResourceTiming`，供后续流水分析消费。

官方公式还涉及整机能力向执行范围的换算及利用率参数，不能只写成不分范围的“bytes / 整机带宽”。命中率模型需要容量；时间换算还需要带宽，这两组配置都必须保持来源明确。

当前这条访存组合使用官方 OpGroup 的最大资源时间作为完成时间，各资源偏移为零。这是明确的服务时间组合假设，不是逐次 DDR→L2→SMEM 的串行往返延迟。Timing 的完成时间和资源时间也不能再次相加。

## 9. 可运行示例：规则 GEMM 的 A/B 读取

以下示例使用官方 H200 架构表，但只展示 A/B 读取的缓存分析。不含 C 写回、SMEM、计算与流水，因此不是完整 GEMM latency 预测，也不用于宣称精度。相联度取 8 是示例模型配置，不是新测得的 H200 硬件事实。

<!-- runnable-cache-example -->
```python
from dataclasses import replace
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.tilesight_new_api.cache import (
    CacheAccessIR, CacheLevelConfig, CacheProblem,
    PanelTraversal, Projection, ReductionConfig, SamplingConfig,
    TensorTileRegion, TileGrid, model_cache,
)

arch = H200_SXM()
arch.set_to_microbench()
M = N = K = 8192
BM = BN = 128
BK = 64
element_bytes = 2

a = TensorTileRegion("A", Projection((0,)), (BM, BK), element_bytes)
b = TensorTileRegion("B", Projection((1,)), (BK, BN), element_bytes)

problem = CacheProblem(
    grid=TileGrid((M // BM, N // BN), axes=("m", "n")),
    accesses=(CacheAccessIR.load("load_a", a), CacheAccessIR.load("load_b", b)),
    traversal=PanelTraversal(row_panel=8, sm_count=arch.sm_count),
    l2=CacheLevelConfig(
        name="l2", capacity_bytes=arch.l2_capacity,
        unit_bytes=min(a.allocation_bytes, b.allocation_bytes),
        associativity=8, cacheline_bytes=128,
        distance_unit="tile_allocation",
    ),
    inner_iterations=K // BK,
    reduction=ReductionConfig(mode="anonymous_inner"),
    sampling=SamplingConfig(seed=0, sample_budget=None),
)

result = model_cache(problem)
for name in ("load_a", "load_b"):
    access = result.by_name(name)
    assert access.request_count > 0
    print(name, {
        "requests": access.request_count,
        "l2_conditional_hit": access.l2_hit_rate_of_l1_5_misses,
        "ddr_bytes_per_request": access.traffic.ddr_read_bytes / access.request_count,
    })
print(result.diagnostics)

# 只改变 panel 配置，比较遍历假设下的缓存结果。
alternative = replace(problem, traversal=PanelTraversal(
    row_panel=16, sm_count=arch.sm_count,
))
alternative_result = model_cache(alternative)
print("aggregate L2 hit:", result.aggregate.l2_hit_rate,
      alternative_result.aggregate.l2_hit_rate)
# 不预设 panel 越大命中率必然越高。
```

示例故意只启用 L2，便于阅读条件命中率与服务比例的关系；实际启用 L1.5 时需同时提供该层配置及分组方式。泛化至不能整除的尺寸时，还必须处理尾块有效范围，不能照抄上述整除写法。

## 10. 两种 swizzle 与当前边界

| 项目 | 表达方式 | 当前边界 |
|---|---|---|
| CTA 调度 swizzle | 工作块坐标和 `PanelTraversal` 等遍历配置 | 可影响复用距离；不等于硬件实际 CTA 发出轨迹 |
| shared-memory layout swizzle | 逻辑元素到存储地址的布局关系；官方 `LayoutMapping(kind="swizzle")` 可记录布局转换 | 不是自动 bank-conflict 成本分析入口 |
| shared-memory 访问成本 | 字节需求及官方 SMEM 带宽 | 不因此获得 lane 地址、bank 冲突或 lowering 插入搬运的精确成本 |

在本次核对的固定版本中，未找到“输入各线程 SMEM 地址，自动输出 bank conflict 成本”的正式接口。`LayoutMapping` 主要用于片上交接、布局转换及消除中间搬运的证明，不能据字段名称扩大其能力。

本模块还具有以下限制：

- 不自动遍历 Region 树生成访问模式；任意嵌套循环、分支及多段 Sequence 需要前端转换。
- 不直接消费 PeriodicDAG 的阶段起点，不按其真实时序重新计算命中率。
- 不自动保留独立 `CacheProblem` 调用之间的缓存状态。
- `TensorTileRegion` 不代表任意元素级地址集合；tile 部分重叠、复杂别名需要先处理。
- 取消请求抽样仍保留执行顺序和多轮干扰假设。
- 平均 Phase 时间不能冒充逐实例冷/热变化或实测 Phase 时间。

## 11. 我们的前端需要交付什么

1. 访问对象、索引关系、有效范围及字节数，来自 Python/高层 TIR 语义。
2. 工作块网格、循环次数、能证明的遍历配置，以及无法证明时采用的假设。
3. 缓存和带宽参数的架构来源、配置来源及单位。
4. 访问名到 semantic access、Phase 的稳定映射。
5. 模型原始输出、平均化和时间转换过程；不隐藏近似，不重复计量。

TileSight 负责既有复用距离、命中概率及资源时间计算。我们的责任不是重新实现缓存硬件，而是保证输入含义、统计范围和回填 Phase 的关系正确。

## 12. 源码索引与验证

官方源码位置（相对固定 TileSight checkout）：

| 文件 | 作用 |
|---|---|
| `tilesight/tilesight_new_api/cache/api.py` | `model_cache` 与后端分派 |
| `tilesight/tilesight_new_api/cache/ir.py` | 输入对象、验证与单位 |
| `tilesight/tilesight_new_api/cache/result.py` | 输出及命中率字段 |
| `tilesight/tilesight_new_api/cache/traversal.py` | grid/panel 遍历 |
| `tilesight/tilesight_new_api/cache/tile_reuse_distance.py` | 不同数据的加权复用距离和内层近似 |
| `tilesight/tilesight_new_api/cache/legacy_volume.py` | 兼容距离后端及共用命中概率评估 |
| `tilesight/util/sdcm.py` | 距离到命中概率 |
| `tilesight/tilesight_new_api/cache/adapters.py` | 官方便捷构造与兼容转换 |
| `tilesight/fused_op_pipeline_wave/overlap_analysis.py` | 流量到资源时间 |
| `tilesight/tilesight_new_api/ir.py` | Timing、布局与片上交接描述 |

本项目：缓存输入投影（实现仓库中的 `src/tilesight_for_tilelang/model/lowering/cache_projection.py`）、访存成本接线（实现仓库中的 `src/tilesight_for_tilelang/model/backends/tilesight/memory_costs.py`）。

2026-09-17 在固定 TileSight checkout 的本地 Python 环境完成验证：5 个 Python 代码块通过语法检查，第 9 节完整示例的两个 panel 配置实际执行成功；各访问服务比例之和为 1，文档本地链接存在。未使用 GPU、nvcc、CUPTI 或 NCU。

该简化示例的字节加权 L2 条件命中率：panel=8 为约 91.9597%，panel=16 为约 93.7232%。这些只是省略 C 写回及 L1.5 后的示例模型输出，不是本项目完整 GEMM 的预测结果，更不是实测值。
