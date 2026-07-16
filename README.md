# BCI IV 2a 跨被试二阶 MAML

该项目使用 MOABB 的 `BNCI2014_001` 加载 BCI Competition IV 2a：

- Inner loop：其他7个被试的22通道数据，空间模块为 Transformer；
- Outer loop：轮换 Query 被试的 `C3、Cz、C4`，空间模块为深度卷积；
- 两个分支共享时间卷积、可分离卷积和分类器；
- 最终测试被试使用第一个会话的三通道数据微调，在第二个会话测试。

每个元训练周期中，8个训练被试都会轮流作为 Query，且其全部试次都会
不重复、不遗漏地经过 Outer loop。Support 批次从其余7个被试等量循环抽取，
避免某个被试因样本顺序或批量大小主导 Inner loop。

## 使用 uv 创建环境

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

虚拟环境默认创建在项目根目录的 `.venv/`，该目录不会提交到 Git。

## 下载并检查数据

```bash
uv run python train_meta.py --download-only
```

MOABB 默认将数据保存到：

```text
data/moabb
```

也可以覆盖路径：

```bash
uv run python train_meta.py --download-only --data-root /path/to/moabb
```

## 运行训练

先运行单个 LOSO 折：

```bash
uv run python train_meta.py --test-subject 1 --meta-epochs 20
```

运行全部9折：

```bash
uv run python train_meta.py --test-subject 0 --meta-epochs 20
```

二阶 MAML 的显存开销较高，可以先降低批量：

```bash
uv run python train_meta.py \
  --test-subject 1 \
  --support-batch-per-subject 4 \
  --query-batch-size 32
```

检查点和指标默认保存在 `outputs/`。每折检查点同时包含元训练模型参数和
该测试被试完成三通道微调后的模型参数。
