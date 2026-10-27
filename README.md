# DSIHF-PM

面向 PM-Bench 的前瞻记忆方法与横向比较代码。主方法在现有源码中使用 **REN（Remaining Evidence Needs）** 名称；本仓库作为后续开发入口保留该命名，避免整理时改变实现含义。

这是首版**代码发布**，包含方法实现、适配层、运行/审核入口、依赖版本和离线测试。不包含实验成绩、轨迹、模型响应、缓存、模型权重或真实凭据。

## 方法

| 方法 | 运行参数 | 实现来源 |
| --- | --- | --- |
| REN（主方法） | `ren` | 本项目：已验证意图账本、条件拆解、剩余证据需求与轮询查询 |
| Single | `single` | PM-Bench 原生单智能体 |
| 可选 Heartbeat | `heartbeat` | PM-Bench 原生 optional heartbeat |
| 固定 30 / 60 分钟 Heartbeat | `heartbeat30` / `heartbeat60` | PM-Bench 原生固定间隔 |
| Hier union | `hier` | PM-Bench 原生层级联合查询 |
| PIS | `pis` | 作者原始 intention store / session / 决策解析器 |
| PMA | `pma` | 作者 MemoryAgent 与三类记忆库 |
| RAG | `rag` | 本项目基础向量检索适配 |
| Mem0 | `mem0` | 官方开源 2.2.1、ADD-only v3 实现 |

第三方源码由 [版本锁](configs/upstreams.json) 固定提交下载，不复制实验仓库的历史或结果。旧的自行实现 PIS 草案、未完成的 BudgetPM、临时服务器恢复脚本不属于此发布。

## 快速开始

已在 Python 3.13 独立环境验证；代码要求 Python 3.11 或更新版本。以下命令在仓库根目录执行：

```bash
git clone https://github.com/LifeforWind/dsihf-pm.git
cd dsihf-pm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.lock.txt
python scripts/bootstrap.py
python -m pytest
```

`bootstrap.py` 固定获取 PM-Bench、PIS、PMA、Mem0，并从 PIS 发布包中只提取经过 SHA256 校验的决策解析器。数据和原作者 checkout 均被 Git 忽略；重复执行会检查版本，遇到不同文件或脏 checkout 会停止。

配置非 Jev 模型使用的中转与 REN 的 Jev 凭据：

```bash
cp a-mem/.env.example a-mem/.env
# 编辑 a-mem/.env，填写自己的地址与密钥；不要提交此文件。
```

远程非 Jev 调用只读取 `a-mem/.env` 中的 `OPENAI_BASE_URL` / `OPENAI_API_KEY`，不使用 OpenRouter fallback。`OPENROUTER_API_KEY` 仅供 REN 的 Jev judge。只跑 Single/PIS/RAG 等基线无需 Jev key。

先查看命令，**不会调用模型**：

```bash
python scripts/run.py --method ren --out runs/ren-preview --dry-run
python scripts/run.py --method pis --out runs/pis-preview --dry-run
```

实际执行一周并进行离线审核：

```bash
python scripts/run.py --method ren --profile relay \
  --model gpt-4o-mini-2024-07-18 --out runs/ren-release-week
python scripts/run.py --method single --profile relay \
  --model gpt-4o-mini-2024-07-18 --out runs/single-release-week
```

默认使用 PM-Bench 发布周。生成指定种子场景：

```bash
mkdir -p runs/scenarios
python benchmarks/PMBench/sim/generate_week_v9.py \
  --seed 4000 --out runs/scenarios/seed-4000.json
python scripts/run.py --method pis --scenario runs/scenarios/seed-4000.json \
  --out runs/pis-seed-4000
```

历史主比较使用 4000–4029；此处只提供生成与运行代码，不附带历史成绩，也不将这些种子称为未见测试集。每次运行必须使用新输出目录；失败尝试和缓存保留，不自动重抽答案。

## PMA / Mem0 额外依赖

```bash
pip install -r requirements-memory.lock.txt
pip install --no-deps -e benchmarks/mem0
python -m spacy download en_core_web_sm
```

PMA 从固定 checkout 导入，无需安装其训练/服务框架。Mem0 必须能加载 spaCy 英文模型与 fastembed BM25 资源；缺失时停止，不静默退化。完整锁文件是既有 Linux 实验环境的依赖记录；其他平台需确认 wheel 可用性。RAG 只需基础依赖和中转 embedding 服务。

## 本地模型服务

提供三个显式配置：`--profile qwen9b` 连接 `127.0.0.1:8001/v1`，模型名 `Qwen3.5-9B`；`--profile qwen4b` 连接 `127.0.0.1:8002/v1`，模型名 `Qwen3.5-4B`；`--profile gemma26b` 连接 `127.0.0.1:8003/v1`，模型名 `gemma-4-26B-A4B-it`。服务需自行部署，关闭 thinking，并支持当前结构化输出请求。`--model` 仅控制 relay 配置。本仓库不启动或修改 GPU 服务；embedding 仍走配置的中转，REN judge 仍使用 Jev。

## 代码导航

- [方法说明](docs/METHODS.md)：主方法逻辑、基线适配及比较边界。
- [源码与版本说明](docs/PROVENANCE.md)：本地/服务器核验、整理时的改动范围。
- [第三方说明](THIRD_PARTY.md)：来源、固定版本与许可证边界。
- `a-mem/agentic_memory/`：`evidence_needs.py`、`prospective.py`、`recommendation.py` 是 REN 的通用状态与调度模块；其余为保留来源的 A-mem 原生代码。
- `benchmarks/PMBench/local_tools/`：REN 适配、九基线接入、预算/传输/缓存审核与回归测试。为保持可追溯性，暂保留历史模块名称；对外入口统一为 `scripts/run.py`。
- `scripts/`：固定上游准备、单周运行与代码发布检查。

独立目录从网络获取四个固定上游、全新安装基础环境后，relay / qwen4b / gemma26b 三种配置各通过 214 项离线测试，另有 1 项依赖历史产物的测试按预期跳过。没有模型 API 调用；未重装完整 Mem0 资源或重跑性能评测。新运行的传输、环境、依赖及代码版本必须与其结果一同记录；不能把整理后的版本直接等同于某次历史实验。
