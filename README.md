# Hibiki-Zero (`kyutai/hibiki-zero-3b-pytorch-bf16`) — Hot-Reloadable Colab Demo

Interactive speech-to-speech and speech-to-text translation demo for **Hibiki-Zero (3B)** designed for rapid iteration on **Google Colab GPUs**.

## Project Layout

* `hibiki_demo/engine.py` — Loads `kyutai/hibiki-zero-3b-pytorch-bf16` and `Mimi` once into a persistent `HibikiBundle` and implements the 12.5 Hz (`80 ms` frame) streaming translation loop.
* `hibiki_demo/ui.py` — Gradio application for speaking into your microphone, watching live English text tokens stream, inspecting the 12.5 Hz alignment plot, and listening to the translated audio (plus a stereo original-vs-translation track).
* `hibiki_zero_colab_demo.ipynb` — Thin Google Colab runner notebook that keeps the 3B model loaded in GPU VRAM while hot-reloading code updates via `!git pull` and `importlib.reload()`.
