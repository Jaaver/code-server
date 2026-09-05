"""
Run PRISM on Modal, detached, so nothing has to stay open.

    pip install modal
    modal setup                       # one-time browser login
    modal run --detach modal_app.py   # returns immediately; the job keeps running

Then close the terminal, the browser, the phone, whatever. Check on it with:

    modal app list                    # find the app id
    modal app logs <app-id>

State lives in a Modal Volume, so it survives across runs the same way the Drive
directory does in Colab: re-running resumes from the last finished phase instead of
starting over.

Options:
    modal run --detach modal_app.py --preset quick --gpu L4 --budget 3600
"""
import modal

APP_NAME = "prism"
app = modal.App(APP_NAME)

# Everything PRISM needs. torch comes from the CUDA-enabled wheel index.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "peft>=0.11",
        "accelerate",
        "datasets",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"HF_HOME": "/state/hf", "TRANSFORMERS_VERBOSITY": "error"})
)

# Named volume: model downloads and PRISM's checkpoint/skills/adapter all persist here.
volume = modal.Volume.from_name("prism-state", create_if_missing=True)


@app.function(image=image, gpu="T4", volumes={"/state": volume},
              timeout=60 * 60 * 6, retries=0)
def run_prism(source: str, env: dict):
    """The whole script is shipped as a string argument.

    That deliberately avoids Modal's local-file mounting API, which has changed names
    across versions (copy_local_file -> add_local_file); passing the source means this
    wrapper keeps working regardless of which Modal version is installed.
    """
    import os, traceback
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["PRISM_STATE"] = "/state/prism_state"
    os.environ["PRISM_DRIVE"] = "0"          # no Google Drive out here
    os.makedirs("/state/prism_state", exist_ok=True)
    try:
        exec(compile(source, "prism_colab.py", "exec"), {"__name__": "__main__"})
    except Exception:
        traceback.print_exc()
        raise
    finally:
        volume.commit()                       # make sure progress is durable either way


@app.local_entrypoint()
def main(preset: str = "tiny", gpu: str = "T4", budget: str = "", model: str = "",
         think: str = "", fresh: bool = False):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    source = open(os.path.join(here, "prism_colab.py")).read()
    env = {"PRISM_PRESET": preset}
    if budget: env["PRISM_TIME_BUDGET"] = budget
    if model:  env["PRISM_MODEL"] = model
    if think:  env["PRISM_THINK"] = think
    if fresh:  env["PRISM_FRESH"] = "1"
    fn = run_prism
    if gpu and gpu != "T4":
        try:
            fn = run_prism.with_options(gpu=gpu)
        except Exception:
            print(f"this Modal version cannot change the GPU at call time — edit gpu= in the "
                  f"@app.function decorator to use {gpu}; running on T4.")
    print(f"submitting PRISM (preset={preset}, gpu={gpu}) to Modal ...")
    fn.remote(source, env)
    print("done. If you launched with --detach this returned immediately and the job "
          "is still running; `modal app logs " + APP_NAME + "` to watch it.")
