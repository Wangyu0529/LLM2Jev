# LLM2Jev

[English](README.md)

LLM2Jev 将本地语言模型适配为 Jev 风格的结构化决策模型。它接受运行时定义的 `Choice`、`Score` 和 `Noul` 问题，并返回包含概率的类型化答案。

> LLM2Jev 是一个独立的开源项目，与 Jev 或 TypeSafe 没有关联，也未获得其认可或授权。

## 快速开始

克隆仓库，并安装项目及 Transformers 推理后端：

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
uv sync --extra transformers
```

使用本地 Hugging Face 兼容的因果语言模型运行示例：

```bash
uv run --extra transformers python examples/transformers_inference.py \
  --model-path /path/to/model
```

示例会提交项目支持的三种问题，并将响应输出为 JSON。

## 问题类型

- `Choice`：选择一个选项，并返回概率分布和置信度。
- `Score`：按照有序等级进行评分，并返回加权分数、概率分布和置信度。
- `Noul`：返回条件成立的概率。

## Python API

```python
from llm2jev import JevRequest, LLM2Jev, Noul, TransformersBackend

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

backend = TransformersBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

Transformers 后端会优先使用 CUDA；没有可用 GPU 时自动回退到 CPU。

## SGLang 后端

可选的 SGLang 后端适用于配有受支持 NVIDIA GPU 的 Linux 环境。

```bash
uv sync --locked --python 3.12 --extra sglang
uv run --extra sglang python examples/sglang_inference.py --model-path /path/to/model
```

同一个 `JevRequest` 可以直接交给 `SGLangBackend`：

```python
from llm2jev import LLM2Jev, SGLangBackend

if __name__ == "__main__":
    with SGLangBackend(model_path, engine_kwargs={"mem_fraction_static": 0.4}) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
```

SGLang 会启动工作进程，因此入口需要 `if __name__ == "__main__":` 保护。
上下文管理器会在退出时关闭引擎。`engine_kwargs` 用于传入 SGLang 引擎配置，
例如示例中的 `mem_fraction_static` 可控制 GPU 显存预算。

## System One HTTP API

`llm2jev-serve` 在 SGLang 原生 HTTP 服务上增加 `POST /v1/systemone`。
模型列表、健康检查、鉴权和其他端点仍由 SGLang 提供。

```bash
uv sync --locked --python 3.12 --extra sglang
uv run --extra sglang llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
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
uv run python -m unittest discover -s tests -v
```

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。
