# RepoDoctor Agent - 面向代码仓库的 AI 诊断助手

一个融合 RAG、Function Calling、静态分析与 LoRA 路由实验的代码仓库问答和诊断系统。

[项目简介](#项目简介) · [核心功能](#核心功能) · [快速开始](#快速开始) · [系统架构](#系统架构) · [实验结果](#实验结果) · [API](#api-接口)

---

## 项目简介

RepoDoctor Agent 面向本地代码仓库提供带证据的问答与诊断能力。系统会根据用户问题自主选择工具，在仓库中检索、读取或分析代码，再由大模型结合工具结果生成回答。

项目当前包含：

- 多工具 Agent：基于 OpenAI 兼容的 Function Calling 接口，最多执行 6 轮工具调用。
- 混合代码检索：Dense 向量检索与 BM25 召回，并使用 RRF 融合结果。
- 查询改写与重排：分别生成语义查询和关键词查询，可选 CrossEncoder Reranker。
- 证据约束：追踪工具证据并校验回答中的文件路径和行号引用。
- 仓库诊断：支持 Python AST 静态分析、Traceback 定位与相关代码读取。
- 完整交互链路：FastAPI 后端、Streamlit Web UI 和多轮会话记忆。
- 路由实验：提供数据构建、评测、Qwen2.5-1.5B LoRA 训练及实验报告。

适用场景包括代码定位、实现理解、代码质量分析、错误日志诊断、代码补全、单元测试生成和 Dockerfile 生成。

---

## 核心功能

### 6 个 Agent 工具

| 工具 | 功能 | 典型场景 |
| --- | --- | --- |
| `search_codebase` | Dense + BM25 + RRF 混合检索 | 定位函数、类和功能实现 |
| `analyze_code` | 使用 AST 分析 Python 文件 | 查看函数、类、导入、圈复杂度和潜在问题 |
| `analyze_log` | 解析异常日志并读取报错上下文 | 定位 Traceback 对应的仓库代码 |
| `complete_code` | 读取指定文件与代码区间 | 为代码修改或补全准备上下文 |
| `generate_tests` | 提取目标结构和源码 | 生成 pytest 测试用例 |
| `generate_dockerfile` | 扫描依赖和程序入口 | 生成适合当前仓库的 Dockerfile |

### 检索链路

```text
用户问题
   |
   v
Query Rewriter
   |-- semantic_query --> Dense Retriever (multilingual-e5-small)
   `-- keyword_query  --> BM25 Retriever
                              |
                              v
                         RRF Fusion
                              |
                              v
                  CrossEncoder Reranker (可选)
                              |
                              v
                       带行号的代码证据
```

### Agent 决策流程

```text
用户输入
   |
   v
LLM 判断任务并选择工具
   |
   v
执行工具并记录证据
   |
   +-- 证据不足 --> 继续调用工具（最多 6 轮）
   |
   v
生成带 [文件路径:起始行-结束行] 引用的回答
   |
   v
引用校验 + 会话记忆
```

仓库中的代码和注释只会作为待分析数据，不会被当作 Agent 指令，也不会直接执行未知仓库代码。

---

## 快速开始

### 环境要求

- Python 3.9+
- 建议 8 GB 以上内存
- 首次建立索引时需要下载 Embedding 模型
- 启用 Reranker 时还需要下载 `BAAI/bge-reranker-base`
- CUDA 可选，仅用于加速本地模型和 LoRA 实验

### 1. 克隆并创建环境

```powershell
git clone https://github.com/lu-chunrui/repo-doctor-agent.git
cd repo-doctor-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 当前 Web 服务需要的补充依赖
pip install fastapi uvicorn streamlit pydantic
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

### 2. 配置环境变量

系统使用 OpenAI Chat Completions 兼容接口。下面以 DeepSeek 为例：

```powershell
$env:LLM_API_KEY = "your-api-key"
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-chat"

# 必须包含准备分析的仓库；多个目录使用系统路径分隔符连接
$env:REPO_DOCTOR_ALLOWED_ROOTS = "D:\桌面"

# 可选：替换默认重排模型
$env:RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
```

是否启用 Reranker 由 Web 界面选项或 `/api/v1/index` 请求中的 `enable_reranker` 字段控制。

> `.env` 已加入忽略列表，但当前程序不会自动加载 `.env`。请在启动进程前设置环境变量。

### 3. 启动 FastAPI 后端

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- 健康检查：`http://127.0.0.1:8000/api/v1/health`
- Swagger 文档：`http://127.0.0.1:8000/docs`

### 4. 启动 Streamlit UI

打开另一个 PowerShell 窗口，并设置相同的工作目录：

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\streamlit_app.py
```

浏览器访问 `http://127.0.0.1:8501`，填写待分析仓库路径，先建立索引，再开始对话或分析日志。

> 第一次建立索引会下载模型并生成 Dense、BM25 索引，因此耗时会明显长于后续启动。索引按仓库隔离保存在 `runtime_data/`。

---

## 系统架构

```text
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                      │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼───────────────────────────────┐
│                     FastAPI Service                      │
│      仓库运行时 · 会话隔离 · 内存管理 · 索引管理          │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    RepoDoctor Agent                      │
│  Function Calling · 多轮工具调用 · 会话记忆 · 引用校验    │
└───────────────┬───────────────────────┬──────────────────┘
                │                       │
┌───────────────▼────────────┐  ┌───────▼──────────────────┐
│         Tool Suite         │  │       RAG Pipeline       │
│ Search / AST / Log / Code  │  │ Rewrite / Dense / BM25   │
│ Test / Dockerfile          │  │ RRF / CrossEncoder       │
└────────────────────────────┘  └──────────────────────────┘
```

### 关键模块

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| Agent Core | `agent/core.py` | LLM 客户端、工具循环与错误恢复 |
| Evidence Agent | `agent/citations.py` | 证据追踪和引用校验 |
| Agent Extensions | `agent/extensions.py` | 多轮会话记忆与扩展 Agent |
| Tool Registry | `tools/registry.py` | 6 个工具的 Schema 与执行入口 |
| Log Analysis | `tools/log_analysis.py` | Traceback 解析与诊断流程 |
| Repository | `repository/` | 扫描、切分、搜索和 AST 分析 |
| RAG | `rag/` | Dense、BM25、RRF、查询改写与重排 |
| API | `api/main.py` | FastAPI 服务与仓库运行时管理 |
| Web UI | `ui/streamlit_app.py` | Streamlit 交互界面 |
| Routing Experiments | `experiments/routing/` | 数据处理、评测与 LoRA 训练 |

---

## 实验结果

路由数据集共 210 条样本，按 `126 / 42 / 42` 划分为训练集、验证集和测试集，覆盖 6 个工具以及直接回答场景。

### Prompt 路由

| 实验 | 数据集 | 决策准确率 | 工具准确率 | Schema 合法率 | 参数键覆盖率 | 参数完全一致率 | 工具幻觉率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Prompt | Validation | 97.62% | 97.22% | 100.00% | 97.22% | 63.89% | 0.00% |
| Prompt v1 | Validation | 100.00% | 100.00% | 100.00% | 100.00% | 66.67% | 0.00% |
| Prompt v1 | Test | 100.00% | 100.00% | 100.00% | 100.00% | 75.00% | 0.00% |

### Qwen2.5-1.5B LoRA 路由

| 模型 | 数据集 | 决策准确率 | 工具准确率 | Schema 合法率 | 参数键覆盖率 | 参数完全一致率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B Base | Validation | 73.81% | 50.00% | 78.57% | 50.00% | 25.00% |
| Qwen2.5-1.5B LoRA | Validation | 90.48% | 88.89% | 100.00% | 77.78% | 25.00% |
| Qwen2.5-1.5B LoRA (epoch 4) | Test | 100.00% | 100.00% | 97.62% | 97.22% | 50.00% |

LoRA 测试集推理记录来自 NVIDIA GeForce RTX 4090 D：平均延迟约 `1.10 s/sample`，吞吐约 `23.83 tokens/s`。

> 这些结果来自 42 条均衡测试样本，用于验证当前路由实验链路，不代表复杂真实仓库场景中的通用准确率。原始指标和预测结果位于 `reports/routing/`。

---

## API 接口

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 查看服务及已加载仓库状态 |
| `POST` | `/api/v1/index` | 为指定仓库建立或加载索引 |
| `POST` | `/api/v1/chat` | 进行带仓库证据的多轮问答 |
| `POST` | `/api/v1/analyze-log` | 分析 Traceback 或错误日志 |
| `POST` | `/api/v1/clear-memory` | 清除指定会话记忆 |
| `GET` | `/api/v1/memory` | 查看指定会话的记忆状态 |

建立索引示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{"repository_path":"D:\\path\\to\\repo","rebuild_index":false,"enable_reranker":true}'
```

对话示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"repository_path":"D:\\path\\to\\repo","session_id":"demo","message":"项目在哪里处理异常日志？"}'
```

---

## 项目结构

```text
repo-doctor-agent/
├── agent/                 # Agent 核心、记忆和引用校验
├── api/                   # FastAPI 服务
├── data/
│   ├── routing/           # 路由训练、验证和测试数据
│   └── lora/              # LoRA 格式数据
├── docs/                  # 实验说明
├── experiments/routing/   # 数据构建、评测和 LoRA 训练脚本
├── rag/                   # Dense、BM25、混合检索、改写与重排
├── reports/routing/       # 实验指标和模型预测
├── repository/            # 仓库读取、切分、搜索与静态分析
├── tools/                 # Agent 工具定义与日志诊断
├── ui/                    # Streamlit 前端
├── runtime_data/          # 按仓库生成的运行时索引（不提交）
└── requirements.txt
```

---

## 当前限制

- 静态分析重点支持 Python；其他文本文件可参与检索，但没有语言级 AST 分析。
- Dense 和 Reranker 模型首次使用时需要下载，离线环境需提前准备缓存。
- 当前路由数据规模较小，实验指标主要用于受控对比。
- 代码生成类工具负责收集可信上下文，最终内容仍由配置的大模型生成。

## License

本项目暂未声明开源许可证。如需复用或分发，请先联系仓库作者。
