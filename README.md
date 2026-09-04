# MomentMaker 静态页面

项目后续开发约定见 [前端开发与联调标准](FRONTEND_STANDARD.md)，执行要求见 [AGENTS.md](AGENTS.md)。当前阶段仍为 HTML/CSS；后续联调版使用 React + Vite + TypeScript + Tailwind CSS + Axios，并通过 REST API 与每 2 秒轮询获取任务进度。本次仅更新规范，尚未迁移技术栈或接入后端。

直接在浏览器打开 `index.html` 即可浏览。不需要安装依赖或构建。

## 页面

- `index.html`：广场与灵感作品。
- `upload.html`：本地文件选择、活动描述与样例入口。
- `template.html`：模板、风格、实体物料选择。
- `editor.html`：作品名称、元素选择与固定合成样例。
- `preview.html`：电子成品、吧唧和小票样机。
- `assets/style.css`：共享样式、CSS 插画和响应式布局。
- `需求原文.md`：压缩包中的 PRD，公开版本已移除成员姓名。

## 范围

仅 HTML 和 CSS，无 JavaScript、后端、第三方字体或远程图片依赖。原生表单允许选择文件、输入文字、勾选模板和元素，但不保存数据、不跨页传递状态、不上传文件、不生成图片。预览均为固定排版样例，下载和发布按钮禁用。

保留原型的创作结构，补齐 PRD 中的预览导出页面。草稿和个人中心不在本版范围。

## 验证

使用本机 Edge 浏览器模拟 1440、768、390、320 像素宽度，5 个页面共 20 个组合未发现页面横向溢出。页面链接返回正常，模板单选可以切换。截图保存在本地 `checks/`，不提交仓库。尚未进行真实 iPhone Safari、微信浏览器测试。

## 仓库与隐私

仓库：https://github.com/banana-Hu/super-octo-rotary-phone

仅提交前端源文件及脱敏项目文档。环境变量文件、密钥、个人上传内容、压缩包、临时文件和验证截图不提交。提交前检查暂存内容，勿在前端代码中写入服务端密钥。
