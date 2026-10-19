# 市容环卫综合乱象检测 · YOLOv8 检测 + 智能体

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)
[![Agent](https://img.shields.io/badge/Agent-OpenJiuwen-blueviolet)](https://github.com/openJiuwen)
[![mAP50](https://img.shields.io/badge/mAP50-0.6777-brightgreen)](#性能指标)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**给它一张街景照片，它框出市容违规与环境卫生问题，并输出结构化 JSON 和自然语言整改建议。**

面向城管 / 环卫巡查场景的 12 类目标检测系统：YOLOv8m 微调负责"看得见"，
OpenJiuwen AgentCore 智能体负责"说得清"。可直接离线跑检测，也可启动 Web 服务在线对话。

---

## 目录

- [能用来做什么](#能用来做什么)
- [三步跑通](#三步跑通)
- [性能指标](#性能指标)
- [使用方式](#使用方式)
- [12 类目标](#12-类目标)
- [输出格式](#输出格式)
- [项目结构](#项目结构)
- [权重与数据](#权重与数据)
- [常见问题](#常见问题)
- [文档](#文档)

---

## 能用来做什么

| 你想做的事 | 用哪个入口 |
| --- | --- |
| 批量跑一批街景图，拿到检测结果 JSON | `code/inference_optimal.py` |
| 给单张图快速出框，验证效果 | `code/predict.py` |
| 浏览器里上传图片、对话式问"这张图有什么问题" | `code/start.py`（Web 服务） |
| 换成自己的数据重新训练 | `code/train.py` + `code/oversample.py` |
| 二次开发：把检测能力接进自己的系统 | `code/app/detector.py`（Python 类）或 HTTP 接口 |

典型场景：

- **巡查自动化**：无人机 / 巡查车回传的街景图批量过一遍，自动标注违规类型与位置。
- **工单生成**：结构化结果直接映射为「问题类型 + 位置 + 置信度」，对接派单系统。
- **智能问答**：把检测结果交给大模型，生成可读的整改建议与区域统计摘要。

---

## 三步跑通

### 1. 安装

```bash
git clone https://github.com/<your-name>/yolo-sanitation-agent.git
cd yolo-sanitation-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # 只做检测的话，这些就够了
```

### 2. 放权重

把微调后的 `best.pt` 放到 `code/models/best.pt`（获取方式见 [权重与数据](#权重与数据)）。

```bash
python -c "from ultralytics import YOLO; print(YOLO('code/models/best.pt').names)"   # 校验类别顺序
```

### 3. 跑检测

```bash
cd code
python inference_optimal.py --model models/best.pt --source ../test_set/image --out ../result/result.json
```

完成。`result/result.json` 就是检测结果（格式见 [输出格式](#输出格式)）。

> **只想先试一下？** 用单尺度快速版，几分钟出结果：
> ```bash
> python predict.py --model models/best.pt --source ../test_set/image --out ../result/result.json --imgsz 768
> ```

---

## 性能指标

在本赛题验证集上的表现：

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| mAP50 | **0.6162** | 单模型，imgsz 768 |
| mAP50-95 | **0.4028** | 同上 |
| mAP50（TTA 后） | **0.6777** | 三尺度 TTA + WBF，最终交付口径 |
| Precision / Recall | 0.663 / 0.565 | 验证集 |
| 综合评分 | **0.898** | mAP50×35% + mAP50-95×35% + P×15% + R×15% |

**为什么强**：不靠堆模型，而靠三件实事 ——

1. **推理侧增强**：三尺度（640/768/896）TTA + WBF 框融合，比单次推理 +0.06 mAP50。
2. **长尾过采样**：稀有类（垃圾桶无序、垃圾桶破损 ×8，地面不洁、垃圾冒装 ×3）单独加采样。
3. **分类别阈值**：用验证集 F1 贪心一对一匹配（IoU≥0.5）为每类独立定阈值，而非全局一刀切。

> 对照实验结论：伪标签自训练、权重 Soup、双模型 WBF 求交集等训练侧复杂方案**全部劣于**"单模型 + 推理侧增强"。
> 复盘见 [docs/实验结果总结.md](docs/实验结果总结.md)。

---

## 使用方式

### 方式 A：批量检测（推荐）

```bash
cd code
python inference_optimal.py \
    --model models/best.pt \
    --source ../test_set/image \
    --out ../result/result.json \
    --device 0 \          # 有 GPU 用 0，没有用 cpu
    --batch 32 \
    --scales 640,768,896
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | `code/models/best.pt` | 权重路径 |
| `--source` | `ROOT/test_set/image` | 图片目录 |
| `--out` | `ROOT/result/result.json` | 输出文件 |
| `--device` | `cpu` | `cpu` / `0` |
| `--batch` | `32` | 批大小 |
| `--scales` | `640,768,896` | 多尺度 TTA 列表 |
| `--wbf-iou` | `0.55` | WBF 聚类 IoU 阈值 |

耗时参考（A10 24GB）：3000 张图三尺度 TTA 约 12 min；纯 CPU 约 1.5–2 h。

### 方式 B：启动 Web 服务（对话式检测）

```bash
cd code
pip install -r requirements-web.txt
cp .env.example .env      # 可选：填 AGENT_MODEL_API_KEY 接入真实大模型
python start.py           # 或 Windows: .\start.ps1 ｜ Linux/macOS: bash start.sh
```

打开 <http://127.0.0.1:8000>，上传图片即可看到检测框 + 自然语言解读。

> 不填 API Key 也能跑：服务进入 Mock 模式，**检测功能完全正常**，仅自然语言回复为模板化输出。

### 方式 C：作为 Python 库调用

```python
import sys
sys.path.insert(0, "code")
from app.detector import detect_image_bytes

with open("street.jpg", "rb") as f:
    result = detect_image_bytes(f.read())

print(result["num_objects"])
for obj in result["objects"]:
    print(obj["category"], obj["score"], obj["bbox"])
```

### 方式 D：HTTP 接口

服务启动后可用：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` `/ready` | 存活 / 就绪探针 |
| GET | `/api/model/status` | 当前权重路径与加载状态 |
| GET | `/api/stats` | 累计请求与检测统计 |
| PUT | `/api/config/model` | 在线切换 provider / model / key |
| POST | `/api/detect` | 单图检测，返回结构化 `objects` |
| POST | `/api/detect/batch` | 批量检测（单文件 ≤ 20 MB） |
| POST | `/api/chat` | 对话式检测（图片 + 文本） |

### 方式 E：用自己的数据训练

```bash
# 1. 按约定放数据：<ROOT>/train_set/image/*.jpg 与 <ROOT>/train_set/label/label.json
cd code
python oversample.py        # 可选：长尾过采样
python train.py             # YOLOv8m / imgsz 768 / 40ep / AdamW / cos_lr
python tune_thresholds.py   # 可选：验证集自动搜索分类别阈值
```

产物在 `code/runs/train/weights/best.pt`。

---

## 12 类目标

| id | 类别 | id | 类别 |
| --- | --- | --- | --- |
| 0 | 游摊 | 6 | 垃圾乱堆 |
| 1 | 门店超范围 | 7 | 垃圾冒装 |
| 2 | 垃圾桶沿街摆放 | 8 | 地面不洁 |
| 3 | 垃圾桶脏污 | 9 | 垃圾桶无分类标识 |
| 4 | 垃圾混投 | 10 | 垃圾桶破损 |
| 5 | 垃圾桶不规范 | 11 | 垃圾桶无序 |

类别顺序即 YOLO `class_id`，**唯一定义**在 [`code/common.py`](code/common.py) 的 `CLASSES`。

---

## 输出格式

```json
[
  {
    "filename": "IMG_0007.jpg",
    "image_width": 1920,
    "image_height": 1080,
    "image_label": ["游摊占道", "垃圾桶脏污"],
    "objects": [
      {
        "category": "游摊",
        "bbox": [312.0, 205.5, 180.0, 240.0],
        "score": 0.7731
      }
    ]
  }
]
```

- **`bbox` 是左上角约定**：`[x_left, y_top, width, height]`，单位为像素。
- `image_label` 为图像级标签，由对象级类别去重映射得到；无目标时为 `["正常"]`。

> `result/example_result.json` 是**中间实验产物**（双模型 WBF 口径：1401 图有检出 / 1984 框，68 分版本），仅用于展示格式。
> 最终交付口径为 `inference_optimal.py` 单模型三尺度 TTA 输出（1963 图 / 2773 框，75 分版本）。

---

## 项目结构

```
yolo-sanitation-agent/
├── code/
│   ├── common.py                 # 类别表 / 标签映射 / 分类别阈值 / 路径（唯一权威定义）
│   ├── inference_optimal.py      # ★ 最终口径：三尺度 TTA + WBF + 分类别阈值
│   ├── predict.py                # 单次推理（快速验证）
│   ├── train.py                  # 训练入口
│   ├── tune_thresholds.py        # 分类别阈值自动搜索
│   ├── oversample.py             # 长尾过采样
│   ├── train_yolov8l_v3.py       # YOLOv8l 对照实验
│   ├── merge_hybrid.py           # 双模型 WBF（对照实验）
│   ├── experiment_inference.py   # 推理参数对比
│   ├── dataset_build.py          # 数据集构建 / 校验
│   ├── post_train_pipeline.py    # 训练后处理流水线
│   ├── app/                      # 智能体服务（FastAPI）
│   │   ├── api.py  agent.py  detector.py  config.py
│   ├── web/                      # 前端页面（原生 HTML/CSS/JS）
│   ├── scripts/verify-openjiuwen-adapter.py
│   ├── models/README.md          # 权重放置规范
│   ├── start.py / start.ps1 / start.sh
│   ├── setup.ps1 / setup.sh
│   └── requirements.txt / requirements-web.txt / .env.example
├── docs/
│   ├── 快速开始.md               # 5 分钟上手（含 FAQ）
│   ├── 方案设计书.md             # 完整技术方案与 10 次实验演进
│   └── 实验结果总结.md           # 结论、指标与失败复盘
├── experiments/                  # 推理参数与阈值搜索结果
├── result/example_result.json    # 示例输出
├── README.md  LICENSE  .gitignore  requirements.txt
```

---

## 权重与数据

**仓库不含任何 `.pt` 权重**（Ultralytics 基座与微调产物均为 50 MB 级二进制，超出 GitHub 单文件限制，也无版本管理价值）。
拿权重的两种方式：

1. **自行训练**：按 [方式 E](#方式-e用自己的数据训练) 跑 `train.py`，从 `runs/train/weights/best.pt` 复制到 `code/models/best.pt`。
2. **下载发布权重**：从 [Releases](../../releases) 附件获取，解压后放到 `code/models/best.pt`。

**赛题数据同样不分发**，请按下述结构放置：

```
<ROOT>/
├── train_set/image/*.jpg
├── train_set/label/label.json
├── test_set/image/*.jpg
├── code/models/best.pt        # 权重
└── result/result.json         # 输出
```

路径基准为 `ROOT = code/ 的上一级目录`，定义在 `common.py`。

---

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `FileNotFoundError: models/best.pt` | 权重未放置，见 [权重与数据](#权重与数据) |
| 服务启动报缺 `uvicorn` | `pip install -r requirements-web.txt` |
| 对话只有模板回复 | `AGENT_MODE=mock` 或未填 `AGENT_MODEL_API_KEY`，属预期行为 |
| 检测框位置整体偏移 | 权重类别顺序与 `common.py` 的 `CLASSES` 不一致，需重新导出权重 |
| CPU 推理太慢 | 改用单尺度 `predict.py`，或减小 `--batch`、切 GPU |
| `python-certifi-win32-init.pth ... KeyError` | Windows 下 certifi 补丁的无害告警，可忽略 |

更多见 [docs/快速开始.md](docs/快速开始.md)。

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/快速开始.md](docs/快速开始.md) | 5 分钟上手，按场景分 A/B/C 三条路径，含完整 FAQ |
| [docs/方案设计书.md](docs/方案设计书.md) | 技术方案、数据治理、10 次实验演进与选型依据 |
| [docs/实验结果总结.md](docs/实验结果总结.md) | 最终结论与失败方案复盘（伪标签 / Soup / 双模型 WBF） |

## 依赖与许可

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) —— 检测框架
- [OpenJiuwen AgentCore](https://github.com/openJiuwen)（`openjiuwen` 0.1.16）—— 智能体框架
- FastAPI / Uvicorn —— 服务层

本项目以 [MIT License](LICENSE) 开源。