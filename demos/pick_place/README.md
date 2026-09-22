# LLM2Jev Pick & Place Arm Demo

This is a self-contained MuJoCo arm demo under LLM2Jev. It uses one Jev
request per control step. The model chooses one of eight atomic actions:

- move the TCP along positive or negative X, Y, or Z
- open the gripper
- close the gripper


MuJoCo owns the simulation and rendering, while the demo owns action limits and the success check. 

## Demo video

[Watch the MuJoCo pick-and-place demo](../../assets/mujoco.mp4). The recorded
run uses Qwen3.5-2B on an NVIDIA GeForce RTX 5090 GPU.

## Dependencies

The LLM2Jev HTTP service needs the normal `sglang` extra and a local Hugging Face causal model. Install the additional demo dependencies with:

```bash
uv pip install mujoco imageio imageio-ffmpeg numpy pillow
```

## Fetch the Panda assets

```bash
python demos/pick_place/fetch_panda.py
```

This creates `demos/pick_place/panda/` with `panda.xml`, meshes,
the upstream license, and a checksum manifest. If you downloaded the release
archive manually, extract only its `franka_emika_panda/` directory there.

## Start the local service

```bash
llm2jev-serve \
  --model-path /data/Qwen/Qwen3.5-2B \
  --served-model-name qwen3.5-2b \
  --host 127.0.0.1 \
  --port 30000 \
  --submission all \
  --attention-backend triton
```

## Run the MuJoCo version

```bash
python demos/pick_place/mujoco_demo.py qwen3.5-2b \
  --video runs/pick-place.mp4
```

On a headless Linux machine, select an available MuJoCo renderer first, for example:

```bash
MUJOCO_GL=egl python demos/pick_place/mujoco_demo.py \
  qwen3.5-2b --video runs/pick-place.mp4
```

## Attribution

The Panda model and meshes come from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie). The Cartesian control, contact handling, task state, and recording approach are adapted from  [`RoboJEV`](https://github.com/lykycy123/RoboJEV). 
