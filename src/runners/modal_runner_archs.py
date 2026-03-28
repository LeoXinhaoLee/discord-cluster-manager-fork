# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.
import modal  # pyright: ignore[reportMissingImports]

from modal_runner import app, cuda_image, modal_run_config

# gpus = ["T4", "L4", "L4:4", "A100-80GB", "H100!", "B200"]
gpus = ["A100-80GB", "H100!", "B200"]

# We intentionally raise `ModalRequeueRequest` (from modal_runner.py) on banned GPU form factors.
# This must propagate as an exception for Modal to retry/requeue the call, so we configure retries here.
_REQUEUE_RETRIES = modal.Retries(
    max_retries=5,
    initial_delay=20.0,
    backoff_coefficient=2.0,
    max_delay=60,
)

for gpu in gpus:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(
        gpu=gpu,
        image=cuda_image,
        name=f"run_cuda_script_{gpu_slug}",
        serialized=True,
        timeout=1200,
        retries=_REQUEUE_RETRIES,
    )(modal_run_config)
    app.function(
        gpu=gpu,
        image=cuda_image,
        name=f"run_pytorch_script_{gpu_slug}",
        serialized=True,
        timeout=1200,
        retries=_REQUEUE_RETRIES,
    )(modal_run_config)
