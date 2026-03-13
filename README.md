# UniMapGenStrongBaseline

独立副本项目，目标是把当前最强 baseline 的训练闭环单独抽出来，便于新成员快速理解并开始训练。

这个项目只保留当前 strongest baseline 所需部分：

- AV2/OpenSatMap 对齐数据读取
- DINOv2 satellite encoder
- Qwen map serialization decoder
- semantic init
- train / eval / predict / official metric

刻意不保留：

- state update 研究支线
- PV 分支
- line type / cut points / 多相机融合等实验支线

因此这里的代码改动不会影响原项目：

- 原项目：`/mnt/data/project/jn/UniMapGen`
- 新项目：`/mnt/data/project/zsy/UniMapGenStrongBaseline`

快速入口见 [Quickstart](docs/quickstart.md)。

补充说明见 [Model Overview](docs/model_overview.md)。
