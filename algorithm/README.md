# MomentMaker 人物抠图模块

本目录是项目中独立的 Python 图像算法模块。它接收一张 JPG、PNG 或 WebP 图片，识别其中的人物实例，并为最多 5 个人分别输出透明背景 PNG、预览图和结构化结果。模块不包含前端页面、业务后端、视频处理或模型训练。

## 技术方案

- 人物实例分割：Torchvision Mask R-CNN（COCO `person` 类）
- 图像处理：Pillow + NumPy
- 默认本地推理，不将用户图片上传到第三方服务
- 模型懒加载：基础测试和非推理代码不会下载权重
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

只运行不涉及真实模型的单元测试时，可安装轻量依赖：

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## 命令行使用

```powershell
momentmaker-cutout .\sample.jpg --output .\output
```

常用参数：

```text
--max-people 5        最多输出人数
--confidence 0.70     人物检测置信度阈值
--mask-threshold 0.50 掩膜二值化阈值
--max-side 1920       推理图片最长边限制
--max-pixels 40000000 输入图片总像素上限
--device cpu|cuda     指定推理设备；默认自动选择
--keep-partial-people 保留贴近画面边缘的狭窄残缺人物
```

默认会过滤贴近左边、右边或上边且主体宽度不足画面 12% 的人物片段，减少模板中出现“半个人”的情况。只贴近底边的人物不会因此被删除；如业务需要保留所有检出结果，可使用 `--keep-partial-people`。

命令会向标准输出打印 JSON。退出码为：`0` 成功、`2` 输入无效、`3` 未检出人物、`4` 模型不可用、`5` 部分成功、`6` 后处理或输出失败。

## Python 调用

```python
from momentmaker_cv.pipeline import process_image

result = process_image("group.jpg", "output")
print(result.status, result.people)
```

`process_image` 也接受自定义 `PersonSegmenter`，便于测试或替换模型，不依赖特定前后端框架。

## 输出目录

```text
output/
├── people/
│   ├── person_01.png
│   └── person_02.png
├── preview.png
└── result.json
```

`result.json` 包含处理状态、图片尺寸、人物置信度、人物框、透明图路径、像素面积、耗时和警告。人物框坐标基于最长边缩放后的处理图片。

## 当前边界

- 遮挡严重、人物极小、逆光、舞台烟雾和复杂发丝可能降低质量。
- 当前预览图仅用于算法验收，不代表最终产品模板设计。
- 模块不会自动判断肖像权或图片授权；调用方应只处理获得授权的素材，并制定上传文件的保存与删除策略。
- 仓库不分发模型权重、测试人像或用户图片。上线前需结合具体使用地区与用途复核预训练权重及数据集条款。

## 代码归属说明

本目录为 MomentMaker 的 Python 图像算法工作区，通过独立分支维护，与现有前端文件隔离。许可证见本目录的 `LICENSE`。
