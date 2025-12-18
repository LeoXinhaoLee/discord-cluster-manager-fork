# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.
from modal_runner import app, cuda_image, modal_run_config

gpus = ["A100-80GB"]
for gpu in gpus:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(gpu=gpu, image=cuda_image, name=f"run_cuda_script_{gpu_slug}_daniel_test", serialized=True, timeout=1200)(
        modal_run_config
    )
    app.function(gpu=gpu, image=cuda_image, name=f"run_pytorch_script_{gpu_slug}_daniel_test", serialized=True, timeout=1200)(
        modal_run_config
    )
