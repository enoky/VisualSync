# VisualSync

Loops equal-length dancer clips over a DJ Mix, retiming every clip to a whole number of bars so the cuts land on the beat, and renders a single MP4 with the mix as its soundtrack.

## Run it on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/enoky/VisualSync/blob/main/VisualSync_Colab.ipynb)

Pick a **T4** or **L4** GPU runtime, run the cells in order, and open the `gradio.live` link the last cell prints. A100/H100 runtimes have no video encoder, so they render on the CPU.

## Run it on your own computer

| Interface | Start it with |
| --- | --- |
| Desktop app (Windows) | `run_gui.bat` |
| Web app (any OS) | `pip install -r requirements.txt gradio` then `python gradio_app.py` |
| Command line | `python make_dance_video.py --bpm 124` (see `--help`) |

All three need `ffmpeg` and `ffprobe` on the PATH. They use an NVIDIA GPU (NVDEC + NVENC) by default. Without one, tick CPU in the web app (it picks it automatically) or pass `--cpu-decode --cpu-encode` on the command line.
