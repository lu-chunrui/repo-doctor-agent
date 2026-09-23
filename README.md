# RepoDoctor Agent

面向本地代码仓库的 AI 问答与诊断系统。项目将混合检索、Function Calling、Python AST 静态分析、Traceback 定位和引用校验组合成一条可评测、可复现的 Agent 链路。

[快速开始](#快速开始) · [工作流程](#工作流程) · [核心能力](#核心能力) · [评测结果](#评测结果) · [评测复现](#评测复现) · [API](#api)

## 项目亮点

- **多工具 Agent**：根据问题选择代码检索、AST 分析、日志诊断、代码补全、测试生成或 Dockerfile 生成工具。
- **混合代码检索**：Dense 与 BM25 双路召回，通过 RRF 融合，并支持 Query Rewrite 与 CrossEncoder Reranker。
- **证据约束回答**：记录每次工具调用的源码证据，校验 `[文件路径:起始行-结束行]` 引用，减少无依据回答。
- **端到端评测**：同时评估工具决策、工具选择、参数、检索证据、引用和最终回答支持度。
- **安全边界**：只允许访问配置目录中的仓库，不执行待分析仓库中的未知代码，并限制输入和工具结果大小。
- **完整交互入口**：提供 FastAPI 后端、Swagger API 文档和 Streamlit Web UI。

## 工作流程

```text
用户问题
   |
   v
意图判断与 Query Rewrite
   |
   +-- semantic query --> Dense Retriever --------+
   |                                               |
   `-- keyword query  --> BM25 Retriever ----------+--> RRF
                                                       |
                                                       v
                                             CrossEncoder Reranker
                                                       |
                                                       v
LLM <---- 带文件路径和行号的证据 <---- Agent 工具调用
   |
   v
回答生成 --> 引用校验 --> 多轮会话记忆 --> API / Web UI
```

软熔断作为可选实验功能保留，默认关闭。当前 30 条检索评测中，开启阈值熔断会误伤 1 条相关查询，而 Query Rewrite 已能过滤无关查询，因此默认配置优先保证召回率。

## 核心能力

### Agent 工具

| 工具 | 作用 | 典型问题 |
| --- | --- | --- |
| `search_codebase` | Dense + BM25 + RRF 混合检索 | “QueryRewriter 在哪里实现？” |
| `analyze_code` | 分析 Python 类、函数、复杂度和告警 | “分析 api/main.py 的结构” |
| `analyze_log` | 解析 Traceback 并读取报错上下文 | “这个 KeyError 是怎么产生的？” |
| `complete_code` | 获取指定文件和代码区间 | “补齐这个方法的参数校验” |
| `generate_tests` | 获取目标类或函数的测试上下文 | “为 CitationValidator 生成 pytest 测试” |
| `generate_dockerfile` | 收集依赖、入口和仓库结构 | “为项目生成 Dockerfile” |

### 证据与引用

涉及仓库事实的回答使用以下格式引用源码：

```text
[rag/hybrid.py:94-110]
```

系统会检查文件是否存在、行号是否有效、引用是否位于工具实际读取的证据范围，并忽略代码块中的引用示例。

## 快速开始

### 环境要求

- Python 3.9+
- 建议 8 GB 以上内存
- 首次运行需要下载 Embedding 模型
- 启用 Reranker 时需要下载 `BAAI/bge-reranker-base`

### 1. 创建虚拟环境

```powershell
git clone https://github.com/lu-chunrui/repo-doctor-agent.git
cd repo-doctor-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# 必须包含准备分析的仓库。Windows 多目录使用分号分隔。
REPO_DOCTOR_ALLOWED_ROOTS=D:\project;D:\workspace
REPO_DOCTOR_CORS_ORIGINS=http://127.0.0.1:8501,http://localhost:8501

EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
DEFAULT_REPOSITORY=D:\project\example-repository

# 实验功能，默认关闭
SOFT_FUSE_ENABLED=false
SOFT_FUSE_SIMILARITY_THRESHOLD=0.869
```

配置由 `pydantic-settings` 自动从项目根目录的 `.env` 读取。`LLM_API_KEY` 在程序导入和健康检查阶段不是必填项，真正调用 LLM 时才会校验。

> 不要提交 `.env`。仓库只应保留不含密钥的 `.env.example`。

### 3. 启动后端

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- 健康检查：<http://127.0.0.1:8000/api/v1/health>
- Swagger：<http://127.0.0.1:8000/docs>

### 4. 启动 Web UI

另开一个终端：

```powershell
cd repo-doctor-agent
.\.venv\Scripts\Activate.ps1
python -m streamlit run ui\streamlit_app.py
```

访问 <http://127.0.0.1:8501>，输入允许目录内的仓库路径，先建立索引，再进行问答或日志诊断。

首次索引需要下载模型并生成 Dense、BM25 索引。运行时索引按仓库隔离保存在 `runtime_data/`，不应提交到 Git。

## 评测结果

所有结果均来自仓库内保存的评测数据和脚本。数据规模仍然较小，指标用于验证当前实现和对比方案，不代表任意代码仓库上的通用效果。

### 检索消融

评测集包含 30 条查询，其中 25 条与仓库相关、5 条无关，覆盖标识符、自然语言、中文口语、跨文件、误导关键词和无关问题。

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均延迟 | 无关查询误检率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.28 | 0.52 | 0.64 | 0.4080 | 1.32 ms | 1.00 |
| Dense | 0.44 | 0.72 | 0.80 | 0.5713 | 18.73 ms | 1.00 |
| Hybrid + RRF | 0.48 | 0.72 | 0.88 | 0.6180 | 21.09 ms | 1.00 |
| Dense + Rewrite | 0.60 | 0.76 | 0.92 | 0.7027 | 8.14 s | 0.00 |
| Hybrid + Rewrite | 0.56 | 0.80 | 0.88 | 0.6800 | 8.18 s | 0.00 |
| **Hybrid + Rewrite + Reranker** | **0.68** | **0.88** | **0.96** | **0.7867** | **11.80 s** | **0.00** |

当前默认方案在该评测集上获得最高 Hit@5 和 MRR，但代价是 Query Rewrite 与 Reranker 带来的明显延迟。原始结果见 `reports/retrieval/`。

### 端到端 Agent

| 指标 | 结果 |
| --- | ---: |
| 用例数 | 30 |
| 完成率 | 100% |
| 工具决策准确率 | 100% |
| 工具选择准确率 | 90% |
| 工具参数准确率 | 100% |
| 证据命中率 | 100% |
| 引用准确率 | 95.45% |
| 无关问题假阳性率 | 0% |
| 平均端到端延迟 | 18.38 s |

### LLM-as-Judge

Judge 会结合问题、回答、工具证据和真实引用源码，评估回答是否被证据支持。

| 指标 | 结果 |
| --- | ---: |
| 成功评判 | 30 / 30 |
| 回答支持率 | 93.33% |
| 幻觉率 | 6.67% |
| 平均相关性 | 4.83 / 5 |
| 平均证据支持度 | 4.16 / 5 |
| 平均引用支持度 | 4.27 / 5 |

Judge 本身也可能产生判断偏差，因此该结果用于发现失败案例，不视为绝对真值。完整记录见 `reports/agent_eval/judge_results.json`。

### 路由实验

路由数据集共 210 条样本，按照 `126 / 42 / 42` 划分训练、验证和测试集，覆盖 6 个工具及直接回答场景。

| 模型或方案 | 数据集 | 决策准确率 | 工具准确率 | Schema 合法率 | 参数键覆盖率 |
| --- | --- | ---: | ---: | ---: | ---: |
| Base Prompt | Validation | 97.62% | 97.22% | 100.00% | 97.22% |
| Prompt v1 | Validation | 100.00% | 100.00% | 100.00% | 100.00% |
| Prompt v1 | Test | 100.00% | 100.00% | 100.00% | 100.00% |
| Qwen2.5-1.5B Base | Validation | 73.81% | 50.00% | 78.57% | 50.00% |
| Qwen2.5-1.5B LoRA | Validation | 90.48% | 88.89% | 100.00% | 77.78% |
| Qwen2.5-1.5B LoRA epoch 4 | Test | 100.00% | 100.00% | 97.62% | 97.22% |

LoRA 测试集包含 42 条均衡样本。相关预测、指标和实验说明位于 `reports/routing/` 与 `docs/routing_experiment.md`。

## 评测复现

先启动 API，然后执行 Agent 评测：

```powershell
python -m scripts.evaluate_agent `
  --repository "D:\path\to\repo-doctor-agent" `
  --timeout 300 `
  --rebuild-index
```

运行 LLM-as-Judge：

```powershell
python -m scripts.evaluate_agent_judge --force
```

校验检索数据集并运行全部检索方案：

```powershell
python -m scripts.validate_retrieval_dataset
python -m scripts.run_all_retrieval_evals
```

Rewrite 和 Judge 都会调用配置的大模型，会产生 API 费用；不同时间、模型和机器上的结果可能变化。

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 服务和仓库运行时状态 |
| `POST` | `/api/v1/index` | 建立或加载仓库索引 |
| `POST` | `/api/v1/chat` | 带仓库证据的多轮问答 |
| `POST` | `/api/v1/analyze-log` | Traceback 和错误日志诊断 |
| `POST` | `/api/v1/clear-memory` | 清除指定会话记忆 |
| `GET` | `/api/v1/memory` | 查看会话记忆状态 |

PowerShell 建立索引示例：

```powershell
$body = @{
    repository_path = "D:\path\to\repository"
    rebuild_index = $false
    enable_reranker = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/index" `
  -ContentType "application/json" `
  -Body $body
```

对话示例：

```powershell
$body = @{
    repository_path = "D:\path\to\repository"
    session_id = "demo"
    message = "项目在哪里处理异常日志？"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/chat" `
  -ContentType "application/json" `
  -Body $body
```

## 项目结构

```text
repo-doctor-agent/
├── agent/                 # Agent、会话记忆、证据与引用校验
├── api/                   # FastAPI 服务和运行时管理
├── data/
│   ├── agent_eval/        # 端到端 Agent 评测集
│   ├── retrieval/         # 检索评测集与仓库配置
│   ├── routing/           # 路由实验数据
│   └── lora/              # LoRA 格式数据
├── experiments/routing/   # 路由和 LoRA 实验脚本
├── rag/                   # Dense、BM25、RRF、改写与重排
├── reports/               # 检索、Agent、Judge 和路由报告
├── repository/            # 扫描、切分、搜索与 AST 分析
├── scripts/               # 数据校验和评测脚本
├── tools/                 # Agent 工具定义与日志诊断
├── ui/                    # Streamlit Web UI
├── runtime_data/          # 本地运行时索引，不提交
├── config.py              # 集中配置
└── requirements.txt
```

## 当前限制

- AST 静态分析目前只支持 Python；其他文本文件可以参与检索，但没有语言级结构分析。
- 当前检索集和 Agent 集均为 30 条，结果仍需在更多仓库和盲测数据上验证。
- Query Rewrite 和 Reranker 提升了当前评测效果，但显著增加延迟。
- 代码补全、测试和 Dockerfile 工具负责收集可信上下文，最终生成内容仍需要人工审查。
- `tests/check_*.py` 主要是手动检查脚本，尚未形成完整的自动化单元测试套件。
- LLM-as-Judge 不是人工标注的替代品，其结果可能受 Judge 模型和提示词影响。

## License

当前仓库尚未添加开源许可证。在添加 `LICENSE` 前，默认保留全部权利。
