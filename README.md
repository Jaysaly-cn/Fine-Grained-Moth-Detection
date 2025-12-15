# Fine-Grained-Moth-Detection
Fine-Grained Moth Detection &amp; Classification
这是一个为您量身定制的 `README.md` 文档。它清晰地梳理了项目背景、技术难点解决方案以及完整的工作流。你可以直接将以下内容保存为项目根目录下的 `README.md` 文件。



---



\# Fine-Grained Moth Detection \& Classification System

\# 基于深度学习的细粒度夜蛾标本检测与识别系统



本项目旨在解决\*\*小样本（Few-Shot）\*\*、\*\*高画质\*\*、\*\*多目标\*\*环境下的昆虫标本识别问题。针对夜蛾标本照片中存在的\*\*标尺干扰\*\*、\*\*姿态各异\*\*以及\*\*物种间纹理差异极小\*\*的挑战，我们设计了一套“\*\*旋转目标检测 + 两阶段微调分类\*\*”的深度学习流水线。



---



\## 🚀 项目亮点 (Key Features)



1\.  \*\*抗干扰检测 (Anti-Interference Detection)\*\*:

&nbsp;   \*   采用 \*\*YOLOv8-OBB (Oriented Bounding Box)\*\* 旋转目标检测技术。

&nbsp;   \*   能够精准剔除标本照片中的尺子、标签和泡沫背景，仅保留纯净的虫体。

&nbsp;   \*   自动修正昆虫姿态，将倾斜的标本校正为水平视角，大幅降低分类难度。



2\.  \*\*小样本/长尾分布优化 (Few-Shot Optimization)\*\*:

&nbsp;   \*   针对单一物种样本极少（~10-20张）的情况，设计了 \*\*两阶段解冻 (Two-Stage Unfreezing)\*\* 训练策略。

&nbsp;   \*   引入 \*\*WeightedRandomSampler\*\* 解决类别不平衡问题，确保稀有物种不被模型忽略。

&nbsp;   \*   结合 \*\*iNaturalist\*\* 预训练权重与 \*\*Albumentations\*\* 强力数据增强，提升模型泛化能力。



3\.  \*\*Top-3 辅助鉴定\*\*:

&nbsp;   \*   推理端输出 Top-1 至 Top-3 的预测结果及置信度，为分类学鉴定提供可靠参考。



---



\## 🛠️ 环境依赖 (Requirements)



本项目基于 Python 3.10 和 PyTorch 构建。



\### 1. 创建环境

```bash

conda create -n moth python=3.10 -y

conda activate moth

```



\### 2. 安装依赖

请确保安装了支持 CUDA 的 PyTorch 版本，然后安装其余依赖：

```bash

pip install ultralytics timm albumentations opencv-python pandas scikit-learn tqdm

```



---



\## 📂 项目结构 (File Structure)



```text

moth\_project/

├── dataset\_detection/      # YOLO检测数据集 (自动生成)

├── cropped\_dataset/        # 裁切后的纯虫体数据集 (自动生成)

├── raw\_images/             # \[输入] 原始标本照片存放处

├── inference\_results/      # \[输出] 推理结果保存处

├── classes.npy             # 类别索引文件

├── best\_finetune\_model.pth # 训练好的分类模型权重

├── train\_detector.py       # 脚本：训练YOLO检测器

├── crop\_images.py          # 脚本：批量裁切与校正

├── filter\_data.py          # 脚本：数据清洗与过滤

├── train\_finetune.py       # 脚本：训练分类模型

└── inference.py            # 脚本：推理与可视化

```



---



\## ⚡ 使用指南 (Usage Guide)



整个工作流分为四个阶段，请按顺序执行。



\### 第一阶段：训练检测器 (Detection)



目标：训练一个能忽略尺子、只看虫子的旋转框检测器。



1\.  \*\*数据标注\*\*：

&nbsp;   \*   使用 \[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 工具。

&nbsp;   \*   挑选约 100-150 张代表性图片。

&nbsp;   \*   使用 \*\*Rotated Rect (旋转矩形)\*\* 模式标注，标签统一为 `moth`。

&nbsp;   \*   \*\*注意\*\*：框要紧贴虫体，\*\*坚决避开尺子\*\*。

&nbsp;   \*   导出为 \*\*YOLO OBB\*\* 格式。

2\.  \*\*开始训练\*\*：

&nbsp;   ```bash

&nbsp;   python train\_detector.py

&nbsp;   ```

&nbsp;   \*训练完成后，权重保存在 `moth\_project/train\_obb\_run1/weights/best.pt`。\*



\### 第二阶段：数据清洗与构建 (Data Preparation)



目标：利用训练好的检测器，清洗所有原始数据，生成高质量分类数据集。



1\.  \*\*批量裁切\*\*：

&nbsp;   将所有原始图片（包括未标注的）放入指定文件夹，运行：

&nbsp;   ```bash

&nbsp;   python crop\_images.py

&nbsp;   ```

&nbsp;   \*此步会自动矫正角度、去除背景，生成纯净小图。\*



2\.  \*\*过滤稀有样本\*\*：

&nbsp;   为了保证分类效果，剔除样本数过少（<10张）的物种：

&nbsp;   ```bash

&nbsp;   python filter\_data.py

&nbsp;   ```

&nbsp;   \*生成 `filtered\_dataset.csv` 索引文件。\*



\### 第三阶段：分类模型训练 (Classification Training)



目标：训练能够区分细粒度纹理的 EfficientNet 模型。



此脚本采用 \*\*两阶段微调\*\* + \*\*加权采样\*\* 策略，专攻小样本过拟合问题。



```bash

python train\_finetune.py

```



\*   \*\*Stage 1\*\*: 冻结骨干网络，只训练分类头（15 Epochs）。

\*   \*\*Stage 2\*\*: 解冻骨干网络，使用极低学习率微调特征（35 Epochs）。

\*   \*输出：`best\_finetune\_model.pth` 和 `classes.npy`。\*



\### 第四阶段：推理与可视化 (Inference)



目标：对新图片进行端到端的检测与识别。



\*\*单张图片测试：\*\*

```bash

python inference.py --source /path/to/image.jpg

```



\*\*批量文件夹测试：\*\*

```bash

python inference.py --source /path/to/test\_folder/

```



结果将保存在 `inference\_results` 文件夹中，图片上会绘制检测框及 Top-3 预测物种名。



---



\## ⚙️ 配置说明 (Configuration)



所有脚本的头部都有配置区域（Configuration Area），你可以根据服务器配置进行调整：



\*   `BATCH\_SIZE`: 默认为 16/32。如果显存不足（OOM），请调小此数值。

\*   `IMG\_SIZE`: 默认为 448。高分辨率有助于保留翅膀纹理细节。

\*   `EPOCHS`: 根据数据量调整，通常两阶段共 50 轮即可收敛。



\## 📊 常见问题 (FAQ)



\*   \*\*Q: 为什么分类准确率在第一阶段很低？\*\*

&nbsp;   \*   A: 这是正常的。第一阶段骨干网络被冻结，模型无法适应夜蛾的特殊纹理。第二阶段解冻后，准确率会显著上升。

\*   \*\*Q: 为什么推理时没有检测到虫子？\*\*

&nbsp;   \*   A: 请检查 `inference.py` 中的 `DETECTION\_CONFIDENCE`。如果标本拍摄背景非常复杂，可能需要适当降低阈值。

\*   \*\*Q: 报错 `classes.npy not found`？\*\*

&nbsp;   \*   A: 必须先运行完 `train\_finetune.py`，该文件会在训练开始时自动生成。



---



\*\*Project Maintainer\*\*: \[Your Name/Lab Name]

