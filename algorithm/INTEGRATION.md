# Python 接入契约

本文档说明业务后端如何调用人物抠图模块，不包含 HTTP 接口或后端框架实现。

比赛演示环境应在 `algorithm/` 下使用以下命令安装经过验证的依赖组合：

```powershell
python -m pip install -c constraints-demo.txt -e ".[model]"
```

约束文件用于复现演示环境，不改变 Python 调用接口，也不要求业务后端提交虚拟环境或模型权重。

## 调用入口

```python
from momentmaker_cv import CutoutOptions, process_image

result = process_image(
    input_path="uploads/job-123/source.jpg",
    output_dir="artifacts/job-123",
    options=CutoutOptions(max_people=5),
)

payload = result.to_dict()
```

未显式传入 `segmenter` 时，同一 Python 进程会复用一个默认模型实例，避免每个请求重复加载权重。默认模型内部会串行执行推理，适合单进程 MVP；如果后续需要更高吞吐量，应由部署层按进程或设备分配模型实例。

调用方负责保存上传文件、尽量分配互不冲突的任务目录，并根据自身数据策略删除原图和结果。算法模块负责读取单张图片并在指定目录生成产物。若重复使用同一目录，模块只会清理由自身命名且不再属于本次结果的 `people/person_数字.png`，其他文件不会被删除。

## JSON 契约 1.0

```json
{
  "schema_version": "1.0",
  "status": "success",
  "input_path": "uploads/job-123/source.jpg",
  "output_dir": "artifacts/job-123",
  "original_size": [1920, 1080],
  "processed_size": [1920, 1080],
  "people": [
    {
      "person_id": 1,
      "confidence": 0.987654,
      "source_box": [100, 80, 620, 1050],
      "output_path": "people/person_01.png",
      "pixel_area": 280000
    }
  ],
  "preview_path": "preview.png",
  "manifest_path": "result.json",
  "warnings": [],
  "error": null,
  "timing_ms": {
    "load": 12.4,
    "inference": 7200.1,
    "postprocess": 210.3,
    "export": 180.6,
    "total": 7603.4
  }
}
```

产物路径相对于 `output_dir`，并始终使用 `/` 分隔，调用方可安全地拼接本地路径或转换为静态资源 URL。`input_path` 和 `output_dir` 保留调用时的路径语义。

## 状态处理

| 状态 | 含义 | 建议处理 |
| --- | --- | --- |
| `success` | 至少成功输出一个人物 | 展示或进入模板流程 |
| `no_person` | 没有人物通过质量筛选 | 提示用户更换图片或降低阈值 |
| `invalid_input` | 文件不存在、格式无效或像素超限 | 返回输入错误 |
| `model_error` | 模型未安装、权重不可用或推理失败 | 记录日志并允许重试 |
| `processing_error` | 掩膜后处理或产物输出失败 | 检查资源与目录权限后重试 |
| `partial_success` | 主要图片已输出，但清单等次要产物失败 | 可使用已有产物，同时记录告警 |

不要根据错误文案判断状态；程序逻辑只依赖 `status` 和 `schema_version`。新增可选字段属于兼容更新，删除字段或改变字段含义时应升级主版本号。
