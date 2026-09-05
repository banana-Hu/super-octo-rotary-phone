# 第三方组件与模型说明

本模块只通过包管理器声明依赖，不在仓库中复制第三方源码、模型权重或数据集图片。

| 组件 | 用途 | 上游许可证 |
| --- | --- | --- |
| PyTorch | 模型推理运行时 | BSD-3-Clause |
| Torchvision | Mask R-CNN 实现与预训练权重入口 | BSD-3-Clause |
| transparent-background | InSPyReNet 前景模型适配与权重入口 | MIT |
| InSPyReNet | 可选显著前景分割模型 | MIT |
| NumPy | 数组计算 | BSD-3-Clause |
| Pillow | 图像读写与滤镜 | HPND |
| pytest | 开发测试 | MIT |

上游项目与许可证：

- PyTorch：<https://github.com/pytorch/pytorch>
- Torchvision：<https://github.com/pytorch/vision>
- transparent-background：<https://github.com/plemeri/transparent-background>
- InSPyReNet：<https://github.com/plemeri/InSPyReNet>
- NumPy：<https://github.com/numpy/numpy>
- Pillow：<https://github.com/python-pillow/Pillow>
- pytest：<https://github.com/pytest-dev/pytest>

默认模型为 `MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT`。首次推理时由 Torchvision 下载权重，权重不会提交到本仓库。该模型使用 COCO 类别体系；数据集说明与使用条款见 <https://cocodataset.org/#termsofuse>。

第三方许可证和模型条款可能更新。本文件用于记录当前技术选型，不替代针对具体发布地区、商业模式、图片来源和人物肖像授权的法律审查。
