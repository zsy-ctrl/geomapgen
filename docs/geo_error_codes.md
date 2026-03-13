# Geo Error Codes

这套编号是给你快速回传问题用的。后面如果脚本报错，优先把类似 `GEO-1204` 这样的编号发给我。

## 入口兜底

- `GEO-1000`: 训练入口未分类异常
- `GEO-1001`: 数据检查入口未分类异常
- `GEO-1002`: tokenizer 检查入口未分类异常
- `GEO-1003`: 推理入口未分类异常
- `GEO-1004`: 评估入口未分类异常
- `GEO-0001`: 手动中断

## 配置与路径

- `GEO-1100`: YAML 配置读取失败
- `GEO-1101`: 数据集 split 目录缺失
- `GEO-1102`: 推理时无法定位输入 split 目录
- `GEO-1103`: 配置里缺少任务对应的 `label_relpaths`
- `GEO-1104`: 当前训练模式下没有可训练参数
- `GEO-1105`: 训练输出目录中的 `config_snapshot.yaml` 或 `run_meta.txt` 写入失败
- `GEO-1106`: JSON 输出写入失败
- `GEO-1107`: checkpoint 原子保存失败
- `GEO-1108`: 输出目录创建失败

## GeoTIFF / GeoJSON 读写

- `GEO-1201`: 读取 RGB GeoTIFF 失败
- `GEO-1202`: 读取二值审核 mask 失败
- `GEO-1203`: 读取 tif 元数据失败
- `GEO-1204`: 读取 GeoJSON 失败
- `GEO-1205`: CRS transformer 创建失败
- `GEO-1206`: GeoJSON 坐标投影失败
- `GEO-1207`: GeoJSON 转像素 feature 失败
- `GEO-1208`: 像素 feature 转 GeoJSON 失败

## Schema / Tokenizer

- `GEO-1301`: 字段类型不支持
- `GEO-1302`: 几何类型不支持
- `GEO-1303`: 某任务没有定义属性字段
- `GEO-1304`: 未知任务 schema
- `GEO-1305`: tokenizer 不支持该值类型
- `GEO-1306`: tokenizer 收到未知任务名
- `GEO-1307`: `transformers.AutoTokenizer` 导入失败
- `GEO-1308`: 结构化 token 注册到 Qwen tokenizer 失败

## 模型与训练状态

- `GEO-1401`: `transformers.AutoModelForCausalLM` 导入失败
- `GEO-1402`: LoRA 依赖 `peft` 不可用
- `GEO-1403`: `llm_train_mode` 非法
- `GEO-1404`: Qwen 模型加载失败
- `GEO-1405`: checkpoint 加载失败
- `GEO-1406`: optimizer / scaler / epoch 状态恢复失败
- `GEO-1407`: 强制使用 CUDA，但当前 torch 不可用 CUDA
- `GEO-1408`: `UNIMAPGEN_DEVICE` 配置非法

## 数据集与推理 tile

- `GEO-1501`: 数据集 split 根目录不存在
- `GEO-1601`: 大图 tile 读取或缩放准备失败
- `GEO-1602`: 某个 tile 的生成阶段失败
- `GEO-1603`: 某个 tile 的 token 反解失败

## 建议回传方式

如果报错，直接把下面三项给我：

1. 错误码，比如 `GEO-1602`
2. 报错整行
3. 你执行的命令

这样我基本不需要你手打长日志。
