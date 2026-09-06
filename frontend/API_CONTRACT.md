# 前端 HTTP 接入契约草案

本文件把团队技术文档中已经出现的接口转换为前端类型。它是联调草案，不代表后端接口已经实现。

## 当前前端假设

- API 前缀为 `/api`，文件资源通过 `/files` 访问。
- 上传使用 `multipart/form-data`，同名字段 `files` 可重复提交。
- 上传响应为 `{ "file_ids": ["..."] }`。
- 创建任务提交 `file_ids`、`template`、`style`，可选 `description` 和 `materials`。
- 创建响应为 `{ "task_id": "..." }`。
- 状态接口沿用团队文档中的 `status`、`progress`、`elements` 和 `result_url`。
- 合成提交 `{ "element_ids": ["..."] }`，响应为 `{ "result_url": "..." }`。
- 作品列表暂按 `{ "items": [], "next_cursor": null }` 解析。
- 发布暂按 `{ "task_id": "...", "title": "..." }` 提交。

其中 multipart 字段名、作品列表包装、完整状态枚举、合成参数、发布请求和错误格式尚未由后端确认。后端提供 OpenAPI 后，应先修改 `src/api/types.ts` 和 `src/api/live.ts`，页面不直接适配协议差异。

## 运行模式

- 默认使用 Mock，页面顶部明确显示“演示模式”。
- 设置 `VITE_USE_MOCK=false` 才会调用真实 HTTP 接口。
- `VITE_API_BASE_URL` 设置 API 前缀，默认 `/api`。
- 本地开发可通过 `VITE_API_PROXY_TARGET` 把 `/api` 和 `/files` 代理到后端测试地址。

真实模式不会在请求失败时伪装为成功或静默切换 Mock。是否自动降级需要产品与后端确认后再实现。
