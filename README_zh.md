  <p align="center">
    <img src="assets/llm2jev-banner.jpeg" alt="LLM2Jev" width="100%">
  </p>

# LLM2Jev：将 LLM 转换为 Jev 风格的决策模型

[English](README.md)

LLM2Jev 将本地语言模型适配为 Jev 风格的结构化决策模型。它接受运行时定义的 `Choice`、`Score` 和 `Noul` 问题，并返回包含概率的类型化答案。

> LLM2Jev 是一个独立的开源项目，与 Jev 或 TypeSafe 没有关联，也未获得其认可或授权。

## 安装

克隆仓库：

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
```

在配有受支持 NVIDIA GPU 的 Linux 环境中，推荐使用 SGLang 后端；它也会安装
自身依赖的 Transformers：

```bash
uv sync --extra sglang
```

如果只需要 Transformers 后端：

```bash
uv sync --extra transformers
```

使用 pip 可编辑安装时，选择对应的 extra：

```bash
python -m pip install -e ".[sglang]"
# 或者：python -m pip install -e ".[transformers]"
```

使用 uv 安装后，激活虚拟环境：

```bash
source .venv/bin/activate
```

## 快速开始

使用本地 Hugging Face 兼容的因果语言模型和受支持的 NVIDIA GPU 运行 SGLang 示例：

```bash
python examples/sglang_inference.py \
  --model-path /path/to/model
```

示例会提交项目支持的三种问题，并将响应输出为 JSON。

## 问题类型

- `Choice`：选择一个选项，并返回概率分布和置信度。
- `Score`：按照有序等级进行评分，并返回加权分数、概率分布和置信度。
- `Noul`：返回条件成立的概率。

## Python API

```python
from llm2jev import JevRequest, LLM2Jev, Noul, SGLangBackend

model_path = "/path/to/model"
request = JevRequest(
    state="客户的包裹一直没有送到。",
    model=model_path,
    questions={
        "is_delivery_issue": Noul(
            instructions="这是物流配送问题吗？",
        )
    },
)

if __name__ == "__main__":
    with SGLangBackend(model_path) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.to_dict())
```

SGLang 会启动工作进程，因此入口需要 `if __name__ == "__main__":` 保护。
上下文管理器会在退出时关闭引擎，`engine_kwargs` 可用于传入 SGLang 引擎配置。

## Transformers 后端

在仅安装 Transformers 后端依赖的环境中运行示例：

```bash
python examples/transformers_inference.py --model-path /path/to/model
```

同一个 `JevRequest` 可以直接交给 `TransformersBackend`：

```python
from llm2jev import LLM2Jev, TransformersBackend

backend = TransformersBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

Transformers 后端会优先使用 CUDA；没有可用 GPU 时自动回退到 CPU。

## System One HTTP API

`llm2jev-serve` 在 SGLang 原生 HTTP 服务上增加 `POST /v1/systemone`。
模型列表、健康检查、鉴权和其他端点仍由 SGLang 提供。

```bash
llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key "$LLM2JEV_API_KEY"
```

查看 SGLang 原生模型列表：

```bash
curl http://localhost:30000/v1/models \
  -H "Authorization: Bearer $LLM2JEV_API_KEY"
```

提交 System One 请求：

```bash
curl http://localhost:30000/v1/systemone \
  -H "Authorization: Bearer $LLM2JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "客户的包裹一直没有送到。",
    "model": "local-model",
    "questions": {
      "delivery": {
        "type": "noul",
        "instructions": "这是物流配送问题吗？"
      }
    }
  }'
```

服务复用 SGLang 的启动参数，目前要求使用默认的单 tokenizer HTTP 模式，
且不能启用 `--skip-tokenizer-init`。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。
