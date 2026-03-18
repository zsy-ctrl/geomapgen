# requirements 安装说明

这份说明对应仓库根目录下的：

- [requirements.txt](/c:/DevelopProject/VScode/geomapgen/requirements.txt)

目标是让你直接用：

```bash
pip install -r requirements.txt
```

把当前 `v1.0` 适配链需要的依赖装起来。

## 适用范围
这份 `requirements.txt` 主要覆盖：

- `v1.0` 当前数据集适配脚本
- family manifest 构建
- Stage A / Stage B 数据导出
- rollout 推理
- `llamafactory-cli train` 训练入口

## 推荐安装方式
先进入仓库根目录：

```bash
cd /home/ads/zsy/geomapgen
```

建议先激活你准备好的环境，再安装：

```bash
conda activate zsy
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 为什么 `requirements.txt` 里是 `cu117`
你之前给的环境是：

- 驱动：`470.129.06`
- CUDA：`11.4`

这份依赖里用的是：

- `torch==1.13.1+cu117`

原因是 `PyTorch 1.13` 官方最常用、最好装的 Linux wheel 是 `cu117`。  
在很多 `CUDA 11.4` 驱动环境下，这套 wheel 仍然可以运行。

## 安装后先做自检
不要上来直接跑训练，先检查环境。

### 1. 检查 torch 和 CUDA
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 2. 检查 `Qwen2.5-VL` 运行支持
```bash
python scripts/check_geo_current_v1_env.py --require-qwen25-vl
```

这一步很重要，因为：

- 我已经把我们自己的脚本改成了旧环境优先走 `fp16`
- 但 `Qwen2.5-VL` 本身是否能在你这套 `torch1.13` 环境里导入，仍然要以实际自检结果为准

## 安装完成后怎么开始
如果自检通过，推荐先跑：

### Stage A 单卡
```bash
bash scripts/run_geo_current_v1_stagea_train_1gpu.sh
```

### Stage A 8 卡
```bash
bash scripts/run_geo_current_v1_stagea_train_8gpu.sh
```

## 说明
当前代码已经做了旧环境兼容处理：

- 训练脚本不再强制 `bf16`
- 在 `torch1.13` 这类环境下会自动退到 `fp16`
- rollout 也支持 `--precision auto|fp16|bf16|fp32`

相关脚本：

- [run_geo_current_v1_stagea_train_1gpu.sh](/c:/DevelopProject/VScode/geomapgen/scripts/run_geo_current_v1_stagea_train_1gpu.sh)
- [run_geo_current_v1_stageb_train_1gpu.sh](/c:/DevelopProject/VScode/geomapgen/scripts/run_geo_current_v1_stageb_train_1gpu.sh)
- [rollout_predict_qwen2_5vl_from_geo_current_family_manifest.py](/c:/DevelopProject/VScode/geomapgen/scripts/rollout_predict_qwen2_5vl_from_geo_current_family_manifest.py)

## 最后提醒
如果：

- `pip install -r requirements.txt` 成功
- 但 `python scripts/check_geo_current_v1_env.py --require-qwen25-vl` 失败

那说明问题不在安装命令本身，而在：

- 当前 `transformers`
- 当前 `torch1.13`
- 和 `Qwen2.5-VL` 运行支持之间的组合

这时先把自检输出贴出来，我们再继续往下收。
