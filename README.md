# tilelang4tilesight-doc

TileLang 性能分析工具的技术路线与 TileSight 接口说明。

本仓库介绍如何从 Python / 高层 TIR 提取语义、工作量与依赖，接入 TileSight 的缓存和流水分析，以及如何用运行时观测生成独立报告和联合预测。

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [模型与运行时两条路径的解法](技术路线/model-runtime-solution-cn.md) | 整体方案、语义提取、组件边界、NCU 观测回灌及报告 |
| [TileSight 阶段、前端接口与流水分析](技术路线/tilesight-ir-and-pipeline-guide-cn.md) | Phase、Actor、Region、PeriodicDAG、跨轮依赖及代码示例 |
| [TileSight 缓存模块](技术路线/tilesight-cache-guide-cn.md) | CacheProblem、数据 tile、遍历、命中率与 Phase Timing |

## 版本与范围

- 文档依据的官方 TileSight 版本：`48e4158459bee5df830ae4ab7dda541edaa3dc4d`。
- 适配器实现基线：`b15b4ac`；文档整理于 2026-09-17。
- 当前讨论的主要平台是 NVIDIA H200。
- 文档中的教学时间和缓存示例结果不代表实测精度。
- 本仓库仅发布技术文档，不包含适配器实现、运行时采集程序或实验产物。文中“实现仓库中的路径”用于说明实现位置，不是本仓库文件。

这三篇文档是对当前接口和实现的说明，不承诺覆盖任意 kernel、任意编译器版本或所有硬件行为。
