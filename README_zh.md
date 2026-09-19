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

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

## 许可证

本项目基于 [MIT License](LICENSE) 发布。
