# Geo Loss Design

## 1. 当前 loss 是什么

当前训练使用的是单一的自回归 token 交叉熵。

实现位置：

- [qwen_geo_generator.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/models/qwen_geo_generator.py#L209)
- [train_geo_model.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/train_geo_model.py#L173)

当前做法可以概括成：

1. 图像先过 DINOv2，得到视觉 token
2. 视觉 token 经过 `sat_proj` 映射到 Qwen hidden size
3. 再拼接 prompt token embedding
4. 再拼接目标 map token embedding
5. 只对目标 map token 计算 causal LM next-token cross entropy

也就是：

`输入 = [satellite tokens] + [prompt tokens] + [target map tokens]`

`监督 = 只监督 target map tokens`

具体细节：

- `prefix_labels` 全部置为 `-100`
- `map_labels` 中 padding 位置置为 `-100`
- `AutoModelForCausalLM(..., labels=labels)` 内部自动做 shift

所以当前 loss 本质上是：

`L_current = CrossEntropy(next_token_prediction)`

## 2. 当前 loss 的优点

### 2.1 工程简单

它完全复用 Hugging Face `CausalLM` 默认训练范式，不需要额外匹配器、几何解码器或复杂后处理参与训练。

### 2.2 稳定

对于 LoRA 和全参训练都稳定，恢复 checkpoint 也简单。

### 2.3 与当前 token 化方案天然兼容

当前输出不是自由 JSON，而是结构化 token 序列。  
在这种前提下，token-level cross entropy 是最自然的第一版目标。

### 2.4 变量长度天然支持

不同 tile 里：

- feature 数量不同
- 每条 lane 点数不同
- 每个 polygon 点数不同

序列建模本身就能覆盖，不需要额外 padding 到固定对象数。

### 2.5 训练链路闭环最短

当前方案已经证明能走通：

- `check -> train -> predict -> eval`

因此非常适合做第一版 baseline。

## 3. 当前 loss 的缺点

### 3.1 与最终评估目标不一致

你最终关心的是：

- lane 几何精度
- intersection polygon 精度
- 关键属性是否正确

但当前 loss 只关心“下一个 token 对不对”，它并不直接关心：

- 坐标差 1 个 bin 和差 50 个 bin 的几何差异
- 线段是否连续
- polygon 是否闭合得合理
- 关键属性字段是否比 `Id` 更重要

### 3.2 所有 token 默认同权

当前所有 map token 在交叉熵里权重相同，这会带来两个问题：

1. 重要字段和不重要字段没有区分
2. 几何 token、字符串 byte token、结构 token 全部按同样权重训练

这是明显不合理的。

### 3.3 可视上不可判定的字段会浪费 loss 预算

例如：

- `Id`
- `RoadId`

这类字段通常不能仅靠图像内容稳定推断。  
如果把它们和 `LaneType`、`TurnType`、`IntersectionType` 一样强监督，会浪费模型容量。

如果业务要求“必须输出这些字段”，更合理的做法通常是：

- 保留监督，但降低权重
- 或者后处理阶段再补

### 3.4 几何 token 只按分类对待，没有“空间距离感”

当前 `<x_127>` 预测成 `<x_126>` 和预测成 `<x_5>`，本质上都只是“分类错了”。  
但从几何意义上，这两种错误完全不是一回事。

### 3.5 长序列会稀释关键属性监督

一条 lane 如果有很多坐标点，坐标 token 会很多。  
这会导致少量关键属性 token 的梯度贡献被淹没。

### 3.6 没有拓扑和对象级约束

当前 loss 不直接约束：

- lane 不能乱断
- intersection polygon 不应自交
- 重复 feature 不应过多
- feature 数量不应明显偏移

### 3.7 暴露偏差问题仍然存在

训练时是 teacher forcing，推理时是自回归采样。  
当前 loss 没有专门缓解 exposure bias。

## 4. 设计目标

如果要升级成“属性加权 + 几何辅助 loss”，建议目标是：

1. 保持当前主链路不推翻
2. 继续保留 token 自回归作为主损失
3. 让关键属性字段更有权重
4. 让几何误差有连续空间意义
5. 尽量不引入复杂的对象匹配器
6. 优先做最小侵入版本，再逐步扩展

## 5. 推荐方案总览

建议分两步做，不要一步到位上太复杂的 set loss。

### Phase 1

保留当前主损失，增加：

- `Weighted Token Cross Entropy`
- `Coordinate Regression Auxiliary Loss`

这是最推荐先落地的一版。

### Phase 2

再增加对象级几何辅助损失：

- `Lane Shape Loss`
- `Intersection Boundary Loss`

## 6. Phase 1: 属性加权 loss

### 6.1 核心思想

不是所有 token 一样重要。  
我们希望按 token 角色和字段名给不同权重。

例如：

- 结构 token：低权重
- `Id` / `RoadId`：低权重
- `LaneType` / `TurnType` / `IntersectionType`：高权重
- `bool` 类型关键字段：中高权重
- 坐标 token：中等或中高权重

### 6.2 建议的 loss 形式

把当前的平均交叉熵改成逐 token 权重版：

`L_wce = sum_i (w_i * CE_i * valid_i) / sum_i (w_i * valid_i)`

其中：

- `CE_i` 是第 `i` 个预测 token 的交叉熵
- `valid_i` 表示这个位置不是 `-100`
- `w_i` 是这个位置的权重

### 6.3 推荐权重策略

建议先用下面这组经验值：

- 结构 token：`0.5`
- 属性 key token：`0.75`
- `Id` / `RoadId` 这类视觉不可判定字段：`0.1 ~ 0.25`
- 关键类别字段：`2.0`
- 关键 bool 字段：`1.5`
- 普通 float / int 字段：`1.25`
- 坐标 token：`1.0 ~ 1.5`

### 6.4 当前任务建议的字段权重

#### Lane

- `Id`: `0.1`
- `RoadId`: `0.1`
- `Length`: `0.5`
- `LaneType`: `2.0`
- `TurnType`: `2.0`
- `RoadIDSource`: `0.5`
- `TurnTypeSource`: `0.5`
- `LaneTypeSource`: `0.5`
- `IsThereStopLine`: `1.5`
- `Width`: `1.25`
- `Source`: `0.5`
- `IsIntersectionInLane`: `1.5`
- `IsIntersectionOutLane`: `1.5`
- `IsThereRoadSplit`: `1.5`
- `IsLeftmost`: `1.25`
- `IsRightmost`: `1.25`

#### Intersection

- `Id`: `0.1`
- `IntersectionType`: `2.0`
- `IsRegular`: `1.5`
- `IntersectionSubType`: `2.0`

### 6.5 为什么不建议把 `Id` 权重拉高

因为 `Id` 和 `RoadId` 在绝大多数情况下不是纯视觉可推断的。  
如果业务强行要求预测这些字段，建议：

- 保留监督，确保格式一致
- 但权重显著低于几何和语义字段

否则模型会把大量能力花在“背字符串”而不是“学路和路口”上。

## 7. Phase 1: 几何辅助 loss

### 7.1 最小可行方案

最小侵入版本，不做序列解码后的复杂匹配，而是直接在 teacher forcing 位置上对坐标 token 加连续值回归。

当前坐标 token 是：

- `<x_i>`
- `<y_i>`

它们本质上是离散 bin。  
可以把分类分布转成“期望坐标”，再和 GT 坐标做回归。

### 7.2 建议形式

对每个坐标位置：

1. 从 logits 中截取 `x` 或 `y` token 对应的子词表
2. 做 softmax
3. 计算期望坐标
4. 与 GT 坐标做 `SmoothL1`

例如：

`x_hat = sum_k softmax(logits_x)_k * x_bin_center_k`

`L_x = SmoothL1(x_hat, x_gt)`

`L_y = SmoothL1(y_hat, y_gt)`

总坐标损失：

`L_coord = mean(L_x + L_y)`

### 7.3 这个方案的优点

- 不需要解码整条序列再做匹配
- 不需要 Hungarian matching
- 对当前 teacher forcing 训练流程侵入很小
- 能体现“差 1 个 bin 比差很多个 bin 更轻”

### 7.4 这个方案的局限

- 它还是局部的点级监督
- 不能直接保证整条 lane 连续
- 不能直接保证 polygon 形状合理

但它已经比纯分类 token loss 更接近几何目标。

## 8. Phase 2: 几何形状辅助 loss

如果 Phase 1 跑起来后，发现：

- lane 容易断
- polygon 顶点顺序抖动
- 点的位置噪声大

再加对象级辅助 loss。

### 8.1 Lane Shape Loss

建议 lane 做 3 部分：

1. `Ordered Point SmoothL1`
2. `Endpoint Loss`
3. `Direction Loss`

可写成：

`L_lane_shape = L_point + 0.5 * L_endpoint + 0.25 * L_direction`

其中：

- `L_point`: GT 顺序点和预测顺序点逐点 `SmoothL1`
- `L_endpoint`: 起点终点额外加权
- `L_direction`: 相邻段向量方向一致性

### 8.2 Intersection Shape Loss

建议 polygon 做 3 部分：

1. `Ordered Boundary SmoothL1`
2. `Perimeter / Edge-Length Consistency`
3. `Area Consistency`

可写成：

`L_poly_shape = L_boundary + 0.25 * L_perimeter + 0.25 * L_area`

这样能帮助 polygon 更稳定。

### 8.3 为什么这里可以不做 Hungarian matching

因为当前训练序列本来就是 teacher forcing，GT feature 顺序在输入中是固定的。  
在第一版实现里，可以先按 GT 顺序直接对齐，不一定要马上引入对象匹配器。

更复杂的匹配式设计适合后面“单次联合生成多个对象且顺序不稳定”时再上。

## 9. 总损失建议

推荐先上这一版：

`L_total = 1.0 * L_wce + 0.3 * L_coord`

如果 Phase 2 再加对象级几何损失：

`L_total = 1.0 * L_wce + 0.3 * L_coord + 0.2 * L_lane_shape + 0.2 * L_poly_shape`

更保守一点也可以：

`L_total = 1.0 * L_wce + 0.2 * L_coord`

不要一开始就把几何辅助项权重拉太高，否则会把语言建模主目标冲掉。

## 10. 建议的实现顺序

### 第一步

先做 `Weighted Token Cross Entropy`。

这是收益最大、风险最低的一步。

### 第二步

再做 `Coordinate Regression Auxiliary Loss`。

这一步已经能显著改善几何学习的“连续性认知”。

### 第三步

观察真实训练结果后，再决定是否需要：

- lane shape loss
- polygon shape loss

## 11. 具体该改哪些文件

### 11.1 `unimapgen/geo/tokenizer.py`

需要新增 token 级元信息输出。  
当前 `encode_task_features()` 只返回 token id，不返回 token 角色。

建议新增类似：

- `encode_task_features_with_meta()`

每个 token 额外返回：

- `role`
- `field_name`
- `feature_idx`
- `point_idx`
- `coord_axis`

例如角色可定义为：

- `structure`
- `task`
- `prop_key`
- `prop_type`
- `prop_value_string`
- `prop_value_number`
- `prop_value_bool`
- `coord_x`
- `coord_y`
- `geom_end`
- `obj_end`

### 11.2 `unimapgen/geo/dataset.py`

当前 dataset 只返回：

- `custom_token_ids`

需要额外返回：

- `custom_token_roles`
- `custom_token_field_ids`
- `coord_target_values`

### 11.3 `GeoVectorCollator`

当前 collator 只 pad：

- `prompt_input_ids`
- `map_input_ids`

需要额外 pad：

- `map_token_weights`
- `map_role_ids`
- `coord_reg_targets`
- `coord_reg_masks`

### 11.4 `unimapgen/models/qwen_geo_generator.py`

当前 `forward()` 直接用 `outputs.loss`。  
如果要做加权和辅助 loss，需要改成：

1. 保留 `outputs.logits`
2. 手动计算 unreduced CE
3. 乘 token 权重
4. 计算 `L_coord`
5. 返回 loss breakdown

建议返回：

- `loss_total`
- `loss_wce`
- `loss_coord`
- `loss_lane_shape`
- `loss_poly_shape`

### 11.5 `unimapgen/train_geo_model.py`

训练器需要：

- 使用 `loss_total`
- 在日志里打印各分项 loss
- 写入 `metrics.jsonl`

### 11.6 配置文件

需要在：

- [geo_vector_lora.yaml](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/configs/geo_vector_lora.yaml)
- [geo_vector_full.yaml](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/configs/geo_vector_full.yaml)

新增 `loss` 配置，例如：

```yaml
loss:
  weighted_ce:
    enabled: true
    structure_weight: 0.5
    coord_weight: 1.0
    field_weights:
      lane:
        Id: 0.1
        RoadId: 0.1
        LaneType: 2.0
        TurnType: 2.0
        Width: 1.25
      intersection:
        Id: 0.1
        IntersectionType: 2.0
        IsRegular: 1.5
        IntersectionSubType: 2.0
  coord_aux:
    enabled: true
    weight: 0.3
    loss_type: smooth_l1
  lane_shape_aux:
    enabled: false
    weight: 0.2
  polygon_shape_aux:
    enabled: false
    weight: 0.2
```

## 12. 推荐的第一版实现范围

如果只做一版最值得的升级，我建议只做这两项：

1. `Weighted Token Cross Entropy`
2. `Coordinate Regression Auxiliary Loss`

不要第一轮就做：

- Hungarian matching
- sequence-level RL
- topology graph loss
- polygon differentiable IoU

这些会显著增加调试复杂度。

## 13. 风险和注意事项

### 13.1 不要把 `Id` 完全当成视觉语义

如果真实数据里 `Id` 只是数据库主键，那它的监督权重必须低。

### 13.2 不要让辅助 loss 压过主 CE

当前模型本质上还是“结构化 token 生成”。  
主损失必须仍然是 token CE。

### 13.3 先从 teacher-forced geometry aux 做起

先做坐标位置级回归，比先做解码后 Chamfer 更稳。

## 14. 一句话建议

当前 loss 适合作为 baseline，但不适合作为最终工程版。  
最合理的升级路径是：

先把“所有 token 同权的 CE”改成“字段可配权重的 CE”，再加“坐标回归辅助 loss”，最后再视真实误差模式决定是否上 lane/polygon 的对象级形状损失。
