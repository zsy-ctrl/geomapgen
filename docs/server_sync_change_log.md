# Server Sync Change Log

## 1. 这份文档的用途

这份文档是给服务器上的 Codex 用的同步说明，不是用户使用手册。

目标是让另一台机器上的 Codex 在不看完整聊天记录的情况下，能够快速理解这几轮已经做过的代码改动，并把对应文件同步过去。

## 2. 当前改动的总目标

在现有 `UniMapGenStrongBaseline` 项目中，新增一套面向私有地理数据的训练/推理链路，支持：

- 输入 `GeoTIFF` 影像进行训练和推理
- 训练监督使用：
  - `patch_tif/0.tif`
  - `patch_tif/0_edit_poly.tif`
  - `label_check_crop/Lane.geojson`
  - `label_check_crop/Intersection.geojson`
- 输出两份预测结果：
  - `Lane.geojson`
  - `Intersection.geojson`
- 冻结 DINOv2
- 支持两种训练模式：
  - 只训练模态转化器 + Qwen LoRA
  - 只训练模态转化器 + Qwen 全量参数
- 支持首次训练新建目录、继续训练从指定 checkpoint 恢复
- 支持大图分块训练和大图滑窗推理
- 支持工程化评估和统一错误码

## 3. 已实现的数据假设

这些假设已经写进代码和配置，服务器侧同步时不要改丢：

- GeoJSON 的 CRS 元数据可信，当前是 `CRS84`
- tif 的 CRS 是投影坐标系，用户示例为 `EPSG:32650`
- GeoJSON 坐标需要先重投影到 tif CRS，再映射到像素坐标
- tif 是 3 通道，`band1=R`, `band2=G`, `band3=B`
- tif 像素值范围是 `0-255`
- `0_edit_poly.tif` 和 `0.tif` 完全对齐
- `0_edit_poly.tif` 是二值 mask，没有灰度过渡
- 白区表示人工审核区域
- `Lane.geojson` 和 `Intersection.geojson` 只保留审核过真值
- 当前先支持：
  - `Lane = LineString`
  - `Intersection = Polygon`
- `Intersection` 当前按外轮廓 polygon 处理

## 4. 这次新增或修改的核心能力

### 4.1 新增一套 `geo` 子模块

新增目录：

- `unimapgen/geo/`

职责：

- 定义任务 schema
- 读取 GeoTIFF / GeoJSON / review mask
- 做 CRS 转换和像素映射
- 做大图 tile 切分
- 做训练样本构造
- 做结构化 token 序列化和反序列化
- 做大图推理结果合并
- 做评估指标和后处理
- 做统一错误码

### 4.2 新增面向 Qwen 的地理矢量生成模型

新增文件：

- `unimapgen/models/qwen_geo_generator.py`

职责：

- 复用现有 `SatelliteEncoder`
- 加载 Qwen causal LM
- 把 DINOv2 视觉 token 通过线性层投影到 Qwen hidden size
- 支持 `freeze / lora / full` 三种 LLM 模式
- 只对结构化 map token 部分计算自回归交叉熵
- 推理时限制到“允许的结构化 token”

### 4.3 新增训练、推理、评估、数据检查入口

新增文件：

- `unimapgen/train_geo_model.py`
- `unimapgen/train_geo_lora.py`
- `unimapgen/train_geo_full.py`
- `unimapgen/predict_geo_vector.py`
- `unimapgen/eval_geo_vector.py`
- `unimapgen/check_geo_data.py`
- `unimapgen/tokenize_geo_vector.py`

这些入口都已经接上统一错误边界，报错时优先输出 `GEO-xxxx` 编号。

### 4.4 新增 Ubuntu/Linux 可直接调用的脚本

新增文件：

- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_full_train.sh`
- `scripts/run_geo_eval.sh`
- `scripts/run_geo_predict.sh`

同时也补了 PowerShell 版本：

- `scripts/run_geo_lora_train.ps1`
- `scripts/run_geo_full_train.ps1`
- `scripts/run_geo_eval.ps1`
- `scripts/run_geo_predict.ps1`

### 4.5 新增配置、依赖和文档

新增文件：

- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`
- `requirements-geo.txt`
- `docs/dependencies_and_install.md`
- `docs/chat_memory_log.md`
- `docs/geo_dataset_structure.md`
- `docs/geo_pipeline_changes.md`
- `docs/geo_training_design.md`
- `docs/geo_error_codes.md`
- `docs/server_sync_change_log.md`

## 5. 代码层面的详细改动

### 5.1 `unimapgen/geo/schema.py`

新增任务 schema 体系：

- `FieldSchema`
- `TaskSchema`
- `load_task_schemas`

默认内置两个任务：

- `lane`
- `intersection`

每个任务定义：

- collection name
- geometry type
- prompt template
- max feature 数
- max point 数
- min point 数
- property fields

当前已写死：

- `lane -> linestring`
- `intersection -> polygon`

这个“写死”指的是 schema 约束，不是写死预测文本。

### 5.2 `unimapgen/geo/tokenizer.py`

新增两层 tokenizer：

- `GeoVectorTokenizer`
- `QwenGeoTokenizer`

实现的关键点：

- 不直接生成自由 JSON
- 使用结构化 token 表达 feature、property 和 geometry
- 字符串字段用 UTF-8 byte token 表示
- 坐标使用 `<x_i>`、`<y_i>` 量化
- 支持 token 反解回 feature records
- 支持 grammar 约束下一 token 合法集合
- 支持对新增 token 做 semantic init

这部分是当前方案稳定性的核心之一。

### 5.3 `unimapgen/geo/io.py`

新增能力：

- 读取 RGB GeoTIFF
- 读取二值 review mask
- 读取 raster metadata
- 加载 GeoJSON
- 检测 GeoJSON CRS
- 用 `pyproj` 建立 CRS transformer
- 将 GeoJSON feature 转像素坐标 feature
- 将像素坐标 feature 转回 GeoJSON

实现上已经按用户要求处理：

- GeoJSON 从 `CRS84` 重投影到 tif CRS
- tif band 顺序使用配置中的 `[1, 2, 3]`
- polygon 输出自动闭合首尾点

### 5.4 `unimapgen/geo/geometry.py`

新增或扩展的几何处理能力：

- review mask bbox 提取
- bbox 扩张
- 大图 tile 生成
- tile 与 mask 的交叠统计
- tile 选择策略
- feature 与 bbox 的相交检测
- feature 对 bbox 的裁剪
- 坐标在原图 / 模型输入尺寸之间变换
- 点重采样
- 方形增强

这部分用于：

- 训练阶段基于 review mask 只取审核区域
- 大图训练分块
- 大图推理滑窗

### 5.5 `unimapgen/geo/dataset.py`

新增 `GeoVectorDataset` 和 `GeoVectorCollator`。

训练样本构造方式：

1. 扫描 split 目录
2. 读取 `0.tif`
3. 读取 `0_edit_poly.tif`
4. 按 review mask 生成 tile
5. 对 lane/intersection 分别构造记录
6. 将 GeoJSON 真值裁到 tile 内
7. 将坐标映射到模型输入尺寸
8. 序列化为结构化 token

当前每个样本会按任务拆分成两类记录：

- lane 记录
- intersection 记录

这意味着训练时是“同图两任务分开学习”，不是联合一条序列。

### 5.6 `unimapgen/geo/inference.py`

实现了大图滑窗推理：

- 按配置生成 tile
- 每个 tile 做 lane 和 intersection 两次生成
- 将模型坐标恢复到原图像素坐标
- 仅保留 tile `keep_bbox` 中心区域的结果，减轻边缘重复
- 跨 tile 去重合并

当前的 tile 合并不是拓扑图级融合，而是“中心保留 + 几何去重”的工程实现。

### 5.7 `unimapgen/geo/metrics.py`

实现了评估和后处理：

- lane 去重
- intersection 去重
- lane 评估
- intersection 评估
- review mask 过滤

当前评估脚本使用这里的逻辑对多个样本做聚合。

### 5.8 `unimapgen/geo/pipeline.py`

负责统一装配：

- schema
- custom tokenizer
- qwen tokenizer
- model
- collator
- dataset

同时处理：

- checkpoint 加载
- optimizer/scaler 状态恢复
- 输出目录创建
- checkpoint 原子保存
- JSON 输出保存

### 5.9 `unimapgen/models/qwen_geo_generator.py`

实现的训练逻辑：

- 图像先过 DINOv2 编码
- 视觉 token 过 `sat_proj`
- 拼接 prompt token
- 拼接 map target token
- 用 Qwen 做标准 causal LM next-token prediction

loss 设计：

- prefix 部分全部 `-100`
- 只对 map token 计算交叉熵
- 当前没有额外几何 loss、拓扑 loss、属性辅助 loss

### 5.10 `unimapgen/train_geo_model.py`

共享训练主循环。

支持：

- LoRA 或 full 模式覆盖
- checkpoint 恢复
- optimizer/scaler 恢复
- best/latest checkpoint 保存
- `metrics.jsonl` 记录
- 输出目录按时间戳命名

目录策略：

- 首次训练默认在 `base_output_dir/<timestamp>/` 下创建新目录
- 若 `resume_in_place=true` 且提供 `init_checkpoint`，则在该 checkpoint 所在目录继续写入

### 5.11 `unimapgen/train_geo_lora.py`

功能：

- 调用共享训练器
- 强制 `mode_override="lora"`

训练目标：

- 冻结 DINOv2
- 训练 `sat_proj`
- 训练 Qwen LoRA 参数

### 5.12 `unimapgen/train_geo_full.py`

功能：

- 调用共享训练器
- 强制 `mode_override="full"`

训练目标：

- 冻结 DINOv2
- 训练 `sat_proj`
- 训练 Qwen 全量参数

### 5.13 `unimapgen/predict_geo_vector.py`

支持：

- 单图推理
- 多图目录推理
- 从 split 自动搜图推理
- 输出每个样本一个目录
- 每个样本输出两份 GeoJSON：
  - `Lane.geojson`
  - `Intersection.geojson`

### 5.14 `unimapgen/eval_geo_vector.py`

支持：

- 按 split 扫描样本
- 对每个样本做大图滑窗推理
- 读取真值 GeoJSON
- 可按 review mask 过滤预测
- 输出总体指标和样本级指标

### 5.15 `unimapgen/check_geo_data.py`

用于不跑训练前先检查：

- split 是否存在
- 影像是否存在
- mask 是否存在
- label 是否存在
- GeoJSON feature 数量
- tif 元数据

### 5.16 `unimapgen/tokenize_geo_vector.py`

用于单样本检查：

- GeoJSON 是否能被正确映射到像素坐标
- 是否能被编码成结构化 token
- 输出 token id 和 token 文本

### 5.17 `unimapgen/utils.py`

本轮额外改动：

- 给 YAML 加载失败打上编号
- 给目录创建失败打上编号
- 给强制设备选择失败打上编号

## 6. 配置层面的改动

新增两个配置：

- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`

其中已经包含：

- 数据路径结构
- band 顺序
- mask 阈值
- prompt template
- lane/intersection schema
- DINO/Qwen 路径
- LoRA 配置
- decode 参数
- 训练参数
- 评估参数
- tiling 参数
- postprocess 参数

### 6.1 关键配置字段

数据：

- `data.dataset_root`
- `data.train_split`
- `data.val_split`
- `data.test_split`
- `data.image_relpath`
- `data.review_mask_relpath`
- `data.label_relpaths.lane`
- `data.label_relpaths.intersection`
- `data.band_indices`

训练：

- `train.base_output_dir`
- `train.output_dir`
- `train.run_name`
- `train.resume_in_place`
- `train.init_checkpoint`
- `train.resume_optimizer`
- `train.resume_scaler`

大图分块：

- `tiling.train.*`
- `tiling.eval.*`
- `tiling.predict.*`

后处理：

- `postprocess.line_dedup_distance_m`
- `postprocess.polygon_dedup_iou`

## 7. 大图裁剪和滑窗改动

这是聊天中后续追加的重要改动，服务器侧要特别注意同步。

### 7.1 训练阶段

不是把整张 tif 直接缩到 `image_size`。

现在的逻辑是：

- 先按 `0_edit_poly.tif` 白区生成 tile
- 再从 `0.tif` 按 tile 窗口读取
- 再把 GeoJSON 真值裁到 tile 内
- 最后缩放到模型输入尺寸

### 7.2 推理阶段

不是整图一次性缩放推理。

现在的逻辑是：

- 整图滑窗
- tile 级生成
- 回到原图坐标
- 去掉 tile 边缘重复
- 去重合并

## 8. 提示词与生成设计

这是聊天中专门解释过的内容，代码已经按这个方向实现：

- lane 和 intersection 分开生成
- 每种任务用一段短 prompt
- prompt 不承担 schema 全约束
- schema 和 grammar 才是主要约束
- 不是自由生成 JSON，而是生成结构化保留 token

详细解释已写进：

- `docs/geo_training_design.md`

## 9. 统一错误码系统

这是聊天最后追加的重要改动。

新增：

- `unimapgen/geo/errors.py`
- `docs/geo_error_codes.md`

现在这些脚本都会优先打印错误码：

- 训练
- 推理
- 评估
- 数据检查
- tokenization 检查

已接入的高频错误场景包括：

- YAML 读取失败
- split 目录缺失
- GeoTIFF 读取失败
- GeoJSON 读取失败
- CRS transformer 失败
- checkpoint 加载失败
- optimizer/scaler 恢复失败
- 某个 tile 的生成失败
- 某个 tile 的 token 反解失败

## 10. 需要同步到服务器的文件清单

下面这些文件是本轮聊天明确新增或修改过、服务器侧需要对齐的最小集合。

### 10.1 必须同步的代码文件

- `unimapgen/geo/__init__.py`
- `unimapgen/geo/dataset.py`
- `unimapgen/geo/errors.py`
- `unimapgen/geo/geometry.py`
- `unimapgen/geo/inference.py`
- `unimapgen/geo/io.py`
- `unimapgen/geo/metrics.py`
- `unimapgen/geo/pipeline.py`
- `unimapgen/geo/schema.py`
- `unimapgen/geo/tokenizer.py`
- `unimapgen/models/qwen_geo_generator.py`
- `unimapgen/train_geo_model.py`
- `unimapgen/train_geo_lora.py`
- `unimapgen/train_geo_full.py`
- `unimapgen/predict_geo_vector.py`
- `unimapgen/eval_geo_vector.py`
- `unimapgen/check_geo_data.py`
- `unimapgen/tokenize_geo_vector.py`
- `unimapgen/utils.py`

### 10.2 必须同步的配置

- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`

### 10.3 建议同步的脚本

- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_full_train.sh`
- `scripts/run_geo_eval.sh`
- `scripts/run_geo_predict.sh`
- `scripts/run_geo_lora_train.ps1`
- `scripts/run_geo_full_train.ps1`
- `scripts/run_geo_eval.ps1`
- `scripts/run_geo_predict.ps1`

### 10.4 建议同步的文档

- `docs/dependencies_and_install.md`
- `docs/chat_memory_log.md`
- `docs/geo_dataset_structure.md`
- `docs/geo_pipeline_changes.md`
- `docs/geo_training_design.md`
- `docs/geo_error_codes.md`
- `docs/server_sync_change_log.md`

### 10.5 建议同步的依赖文件

- `requirements-geo.txt`

## 11. 尚未完成的验证

这点必须同步告诉服务器上的 Codex。

当前环境里没有可执行的 `python` / `py` / `wsl python3`，所以这些改动是：

- 已完成源码级实现
- 未在当前本地沙箱做真实运行验证

换句话说，这一批代码不是“已经在本机跑通”，而是“已经按需求完成改造，但需要在服务器 Ubuntu 环境做第一轮联调”。

## 12. 服务器侧建议优先验证顺序

服务器上的 Codex 建议按这个顺序验证：

1. 安装 `requirements-geo.txt`
2. 跑 `check_geo_data.py`
3. 跑 `tokenize_geo_vector.py`
4. 跑 `train_geo_lora.py`
5. 跑 `predict_geo_vector.py`
6. 跑 `eval_geo_vector.py`

如果出现报错，优先根据：

- `docs/geo_error_codes.md`

按 `GEO-xxxx` 定位问题。

## 13. 服务器上的 Codex 需要特别注意的点

### 13.1 不要回退原有 opensatmap 路线

这次新增的是一条独立的 `geo` 训练/推理链路，不是替换原有 `opensatmap` 主线。

### 13.2 当前支持的几何类型有限

第一版默认只处理：

- lane 的 `LineString`
- intersection 的 `Polygon`

如果服务器上发现真实数据里存在：

- `MultiLineString`
- `MultiPolygon`
- polygon 洞
- 空几何

则需要继续扩展 `io.py`、`schema.py`、`tokenizer.py`。

### 13.3 当前 loss 只有 token-level 交叉熵

服务器侧如果发现：

- 属性预测不稳
- 长中心线断裂
- 路口 polygon 顶点不稳

下一步优先考虑增加辅助 loss，而不是推翻当前 token 生成框架。

### 13.4 大图 merge 目前是工程版，不是拓扑版

当前跨 tile 合并主要依赖：

- keep bbox
- 中心区域保留
- 几何去重

不是图结构级的拓扑融合。

## 14. 一句话总结

本轮聊天已经把项目从原始 baseline 扩展出一套独立的 `GeoTIFF + Lane/Intersection GeoJSON` 训练、推理、评估、大图分块和错误码体系；服务器上的 Codex 需要以本文件列出的文件集合为准完成同步，并在 Ubuntu 环境完成第一轮运行验证。
