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
from llm2jev import JevRequest, LLM2Jev, Noul, TransformersBackend

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

backend = TransformersBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

The Transformers backend uses CUDA when available and otherwise falls back to CPU.

## SGLang Backend

The optional SGLang backend runs on Linux with a supported NVIDIA GPU.

```bash
uv sync --locked --python 3.12 --extra sglang
uv run --extra sglang python examples/sglang_inference.py --model-path /path/to/model
```

Use `SGLangBackend` with the same `JevRequest`:

```python
from llm2jev import LLM2Jev, SGLangBackend

if __name__ == "__main__":
    with SGLangBackend(model_path, engine_kwargs={"mem_fraction_static": 0.4}) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
```

Use a main guard because SGLang launches worker processes. The context manager
shuts down the engine on exit. Pass SGLang engine options through `engine_kwargs`;
the example sets the GPU memory budget with `mem_fraction_static`.

## System One HTTP API

`llm2jev-serve` adds `POST /v1/systemone` to SGLang's native HTTP server.
SGLang continues to provide model listing, health checks, authentication, and
its other native endpoints.

```bash
uv sync --locked --python 3.12 --extra sglang
uv run --extra sglang llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
  --api-key "$LLM2JEV_API_KEY"
```

List models through SGLang's native endpoint:

```bash
curl http://localhost:30000/v1/models \
  -H "Authorization: Bearer $LLM2JEV_API_KEY"
```

Submit a System One request:

```bash
curl http://localhost:30000/v1/systemone \
  -H "Authorization: Bearer $LLM2JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The customer package has not arrived.",
    "model": "local-model",
    "questions": {
      "delivery": {
        "type": "noul",
        "instructions": "Is this a delivery issue?"
      }
    }
  }'
```

The command accepts SGLang's normal server arguments. It currently requires
the default single-tokenizer HTTP mode and does not support
`--skip-tokenizer-init`.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
