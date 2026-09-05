# 主体组合抠图验证记录

## 环境

- Windows，Python 3.14，CPU 推理
- `torchvision` Mask R-CNN V2
- `transparent-background` 1.3.4 / InSPyReNet
- 处理图片最长边 1920 像素

## 测试素材

测试图片仅保存在本地临时目录，未提交到仓库。

- [两人共同使用 Framework Laptop 13](https://commons.wikimedia.org/wiki/File:2_people_using_Framework_Laptop_13_with_DC-ROMA_RISC-V_Mainboard_inside,_at_Maker_Faire_Rome_2025.jpg)，Wikimedia Commons 页面标注 CC BY-SA 4.0。
- [Woman holding a camera](https://commons.wikimedia.org/wiki/File:Woman_holding_a_camera.jpg)，Wikimedia Commons 页面标注 CC0。
- [Robotics Challenge](https://commons.wikimedia.org/wiki/File:Robotics_Challenge_(5191842).jpg)，Wikimedia Commons 收录的美国陆军图片；本次只做本地技术验证，不再分发素材。

## 结果

- 笔记本图片：检出 2 人并组成 1 个主体；透明结果保留两人、手部和笔记本，主要背景被移除。该结果符合模板示例所需的“人物组合 + 交互物体”。
- 相机图片：检出 1 人并生成 1 个主体；相机和与人物连接的相机包被保留。
- 机器人图片：前景人物与较小的背景人物不再错误合并；前景人物主体正确，但桌面机器人没有被稳定保留。该图片验证了当前方法对“靠近但未形成明显前景连接”的物体没有保证。

三张图片均返回 `success`，逐人 PNG 和主体 PNG 均正常写入。首次包含模型初始化的单张总耗时约 19–32 秒；后续耗时取决于图片尺寸和设备。比赛前必须预下载权重，并以实际演示机复测。

## 结论

当前实现满足黑客松 MVP：保留原有逐人结果，近距离人物可合并，典型手持或共同使用物体可随主体输出，并具备失败降级。它不是通用关系理解模型；机器人等未形成清晰连接的物体仍应列为已知限制。
