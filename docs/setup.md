# 本地启动

## 环境

- Python 3.12
- Node.js 24；前端支持版本以package.json为准
- PostgreSQL 16
- Redis
- Milvus 3.0.5
- DashScope API Key

## 1. 基础服务

准备PostgreSQL、Redis和Milvus，使用以下本地连接配置：

| 服务 | 主机端口 | 配置 |
|---|---:|---|
| PostgreSQL | 15432 | 创建rag_eval数据库 |
| Redis | 16379 | 数据库0 |
| Milvus | 19530 | 配合对应版本的etcd、MinIO部署 |

## 2. 后端

```powershell
cd backend
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
./.venv/Scripts/python.exe run.py
```

在.env中填写DashScope Key、数据库账号和密码，并确认三项服务连接地址。

- 健康检查：http://127.0.0.1:8000/health
- 接口文档：http://127.0.0.1:8000/docs

OCR配置项位于backend/app/config.py，配置相应服务地址和认证信息后开启OCR_ENABLED。

## 3. 前端

```powershell
cd front/rag_front
npm ci
npm run dev
```

访问终端输出的开发地址。Vite将/api/v1和/health代理到127.0.0.1:8000。

## 4. 使用流程

上传文档或录入文本 → 查看文档和Chunk → 检索调试 → Agent对话 → 查看来源。
