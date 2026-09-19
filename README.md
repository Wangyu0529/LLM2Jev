# LLM2Jev

[简体中文](README_zh.md)

LLM2Jev adapts local language models to Jev-style structured decisions. It accepts runtime-defined `Choice`, `Score`, and `Noul` questions and returns typed answers with probabilities.

> LLM2Jev is an independent open-source project. It is not affiliated with or endorsed by Jev or TypeSafe.

## Quick Start

Clone the repository and install the project with its Transformers backend:

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
uv sync --extra transformers
```

Run the example with a local Hugging Face-compatible causal language model:

```bash
uv run --extra transformers python examples/transformers_inference.py \
  --model-path /path/to/model
```

The example submits all three supported question types and prints the response as JSON.

## Question Types

- `Choice`: selects one option and returns a probability distribution and confidence.
- `Score`: evaluates ordered levels and returns a weighted score, probability distribution, and confidence.
- `Noul`: returns the probability that a condition is true.

## Python API

```python
from llm2jev import JevRequest, LLM2Jev, Noul, TransformersBinaryBackend

model_path = "/path/to/model"
request = JevRequest(
    state="The package has not arrived.",
    model=model_path,
    questions={
        "is_delivery_issue": Noul(
            instructions="Is this a delivery issue?",
        )
    },
)

backend = TransformersBinaryBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

The Transformers backend uses CUDA when available and otherwise falls back to CPU.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## License

This project is licensed under the [MIT License](LICENSE).
