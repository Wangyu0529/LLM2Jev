  <p align="center">
    <img src="assets/llm2jev-banner.jpeg" alt="LLM2Jev" width="100%">
  </p>

# LLM2Jev: Turn LLMs into Jev-Style Decision Models

[简体中文](README_zh.md)

LLM2Jev adapts local language models to Jev-style structured decisions. It accepts runtime-defined `Choice`, `Score`, and `Noul` questions and returns typed answers with probabilities.

> LLM2Jev is an independent open-source project. It is not affiliated with or endorsed by Jev or TypeSafe.

## Installation

Clone the repository:

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
```

SGLang is the recommended backend on Linux with a supported NVIDIA GPU. It also
installs its Transformers dependency:

```bash
uv sync --extra sglang
```

For a Transformers-only environment:

```bash
uv sync --extra transformers
```

For an editable pip installation, use the corresponding extra:

```bash
python -m pip install -e ".[sglang]"
# Or: python -m pip install -e ".[transformers]"
```

After installing with uv, activate the virtual environment:

```bash
source .venv/bin/activate
```

## Quick Start

Run the SGLang example with a local Hugging Face-compatible causal language model and a supported NVIDIA GPU:

```bash
python examples/sglang_inference.py \
  --model-path /path/to/model
```

The example submits all three supported question types and prints the response as JSON.

## Question Types

- `Choice`: selects one option and returns a probability distribution and confidence.
- `Score`: evaluates ordered levels and returns a weighted score, probability distribution, and confidence.
- `Noul`: returns the probability that a condition is true.

## Python API

```python
from llm2jev import JevRequest, LLM2Jev, Noul, SGLangBackend

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

if __name__ == "__main__":
    with SGLangBackend(model_path) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.to_dict())
```

Use a main guard because SGLang launches worker processes. The context manager
shuts down the engine on exit. Pass SGLang engine options through `engine_kwargs`.

## Transformers Backend

Run the example in the Transformers-only environment:

```bash
python examples/transformers_inference.py --model-path /path/to/model
```

Use `TransformersBackend` with the same `JevRequest`:

```python
from llm2jev import LLM2Jev, TransformersBackend

backend = TransformersBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

The Transformers backend uses CUDA when available and otherwise falls back to CPU.

## System One HTTP API

`llm2jev-serve` adds `POST /v1/systemone` to SGLang's native HTTP server.
SGLang continues to provide model listing, health checks, authentication, and
its other native endpoints.

```bash
llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
  --host 0.0.0.0 \
  --port 30000 \
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
python -m unittest discover -s tests -v
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
