# MomentMaker 人物抠图模块

本目录是项目中独立的 Python 图像算法模块。它接收一张 JPG、PNG 或 WebP 图片，为最多 5 个人分别输出透明背景 PNG，并额外输出可放入模板的近距离人物组合。可选前景增强模式会尽量保留人物手持、怀抱或紧密连接的物体。模块不包含前端页面、业务后端、视频处理或模型训练。

## 技术方案

- 人物实例分割：Torchvision Mask R-CNN（COCO `person` 类）
- 可选前景增强：InSPyReNet 显著前景分割
- 图像处理：Pillow + NumPy
- 默认本地推理，不将用户图片上传到第三方服务
- 模型懒加载：基础测试和非推理代码不会下载权重
- 同一 Python 进程会复用默认模型实例，并串行保护模型初始化和推理
- 模型适配器可替换，后续可接入其他本地模型或 MCP/API

选择 Mask R-CNN 是因为它能区分同一张图片中的多个人物，Torchvision 项目使用 BSD-3-Clause 许可证，并且 Windows 安装路径相对清晰。CPU 可以运行，但首次启动需要下载模型权重，推理速度取决于设备。

## 安装

建议使用 Python 3.11–3.14 和独立虚拟环境。

```powershell
cd algorithm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[model,dev]"
```

比赛演示机建议使用仓库中已经验证过的直接依赖版本：

```powershell
python -m pip install -c constraints-demo.txt -e ".[model,dev]"
```

需要保留人物连接物体时，安装可选前景依赖：

```powershell
python -m pip install -c constraints-demo.txt -e ".[model,foreground,dev]"
```

`constraints-demo.txt` 记录的是 Windows + Python 3.14 环境中完成单元测试和真实图片推理时使用的版本。它只锁定本模块直接使用的依赖，不锁定操作系统相关的间接依赖。普通开发仍可使用上方不带约束文件的安装命令。

只运行不涉及真实模型的单元测试时，可安装轻量依赖：

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## 演示机准备与自检

联网安装完成后，提前运行一次模型自检。该命令会下载或读取本地缓存中的权重，并完成一次最小推理：

```powershell
momentmaker-check --device cpu
```

使用前景增强模式时，必须同时预热并检查 InSPyReNet：

```powershell
momentmaker-check --device cpu --subject-mode foreground
```

输出中的 `status` 为 `ready` 才表示模型可以使用。比赛现场没有验证过 CUDA 环境时，建议明确使用 `--device cpu`；使用 GPU 前应在实际演示机上将参数改为 `--device cuda` 并完成同样的检查。

准备好一张已获授权、画面中至少有一人的测试照片后，可以执行完整验收：

```powershell
momentmaker-check .\sample.jpg --output .\output\smoke --device cpu --subject-mode foreground
```

完整验收会运行人物分割，并自动检查逐人及主体 PNG 是否为 RGBA、是否包含有效透明区域，以及预览图、清单、人物编号、主体数量和主主体编号是否一致。指定 `foreground` 时还会确认增强模式没有静默降级。测试图片和生成结果仅保留在本地，不提交到仓库。

## 命令行使用

```powershell
momentmaker-cutout .\sample.jpg --output .\output
```

常用参数：

```text
--max-people 5        最多输出人数
--confidence 0.70     人物检测置信度阈值
--mask-threshold 0.50 掩膜二值化阈值
--alpha-mode soft|hard 边缘模式；默认 soft，可回退 hard
--subject-mode none|people|foreground 主体模式；默认 people
--max-side 1920       推理图片最长边限制
--max-pixels 40000000 输入图片总像素上限
--device cpu|cuda     指定推理设备；默认自动选择
--keep-partial-people 保留贴近画面边缘的狭窄残缺人物
```

默认会过滤贴近左边、右边或上边且主体宽度不足画面 12% 的人物片段，减少模板中出现“半个人”的情况。只贴近底边的人物不会因此被删除；如业务需要保留所有检出结果，可使用 `--keep-partial-people`。

系统先按置信度选出最多 5 个有效人物，再按人物在原图中的水平位置从左到右编号，保证模板放置顺序稳定。

`people` 模式根据人物框距离和垂直重叠关系组合近距离人物，不增加模型推理。`foreground` 模式额外保留组合附近且与人物连通的显著前景，适合合照中的笔记本、相机等物体。增强模型失败时自动回退到人物组合，并在 `warnings` 中记录原因。CPU 实测中，增强模型缓存权重后加载约 5 秒、每张约 9–10 秒；比赛前应在演示机完成安装和权重预热。

默认 `soft` 模式使用模型原始概率生成连续 Alpha，同时将过渡范围限制在清理后的主体边缘附近，以减少人工羽化产生的光晕。若特定图片出现边缘异常，可通过 `--alpha-mode hard` 回退到二值掩膜加高斯羽化。

命令会向标准输出打印 JSON。退出码为：`0` 成功、`2` 输入无效、`3` 未检出人物、`4` 模型不可用、`5` 部分成功、`6` 后处理或输出失败。

## Python 调用

```python
from momentmaker_cv import process_image

result = process_image("group.jpg", "output")
print(result.status, result.people, result.subjects)
```

`process_image` 也接受自定义 `PersonSegmenter`，便于测试或替换模型，不依赖特定前后端框架。

供后端使用的稳定字段、状态处理和路径规则见 [INTEGRATION.md](INTEGRATION.md)。

## 输出目录

```text
output/
├── people/
│   ├── person_01.png
│   └── person_02.png
├── subjects/
│   └── subject_01.png
├── preview.png
└── result.json
```

`result.json` 使用版本化 JSON 契约，包含处理状态、图片尺寸、逐人结果、主体组合、耗时和警告。`subjects[].member_person_ids` 表示组合包含的逐人编号，`mode` 表示实际使用 `people` 或 `foreground`。`primary_subject_id` 指向透明像素面积最大的主体，供模板流程直接选择主要元素。所有产物路径均相对于输出目录并使用 `/` 分隔；坐标基于最长边缩放后的处理图片。

重复使用同一个输出目录时，模块会在新结果成功生成后删除 `people/` 和 `subjects/` 下多余的算法产物。清理不递归，也不会删除不符合命名规则的文件；删除失败会记录在 `warnings` 中，不影响本次有效结果。

`preview.png` 是算法验收图，每个人物卡使用浅色和深色各一半的背景，用于检查透明边缘在不同模板底色上的白边、黑边和断层；它不代表最终产品模板。

## 当前边界

- 遮挡严重、人物极小、逆光、舞台烟雾和复杂发丝可能降低质量。
- 当前预览图仅用于算法验收，不代表最终产品模板设计。
- 前景增强不能保证保留所有未与人物接触的交互物体，也可能多保留与人物相连的桌面或包。
- 模块不会自动判断肖像权或图片授权；调用方应只处理获得授权的素材，并制定上传文件的保存与删除策略。
- 仓库不分发模型权重、测试人像或用户图片。上线前需结合具体使用地区与用途复核预训练权重及数据集条款。

## 代码归属说明

本目录为 MomentMaker 的 Python 图像算法工作区，通过独立分支维护，与现有前端文件隔离。许可证见本目录的 `LICENSE`。
