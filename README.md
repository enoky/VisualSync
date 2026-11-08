# VisualSync

Loops equal-length dancer clips over a DJ Mix, retiming every clip to a whole number of bars so the cuts land on the beat, and renders a single MP4 with the mix as its soundtrack.

## Run it on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/enoky/VisualSync/blob/main/VisualSync_Colab.ipynb)

Pick a **T4** or **L4** GPU runtime, run the cells in order, and open the `gradio.live` link the last cell prints. A100/H100 runtimes have no video encoder, so they render on the CPU.

### Send finished videos to YouTube (Colab + IFTTT)

Big videos can't go to YouTube straight from a Google Drive share link (Drive shows a virus-scan page instead of the file). VisualSync works around it without the video touching your phone:

1. Create an empty Google Sheet named **YouTube Queue** in My Drive.
2. Make an IFTTT Applet: Google Sheets *New row added to spreadsheet* (that sheet) → YouTube *Upload video from URL*, with Video URL = `ColumnA`, Title = `ColumnB`, Description = `ColumnC`.
3. In the notebook, leave **SEND_TO_YOUTUBE** ticked in cell 2 (it asks you to sign in to Google).
4. In the app, open **Send to YouTube** and tick *Send each full video…*, or press **Send to YouTube** for any video on Drive. Cell 5 does the same for videos rendered earlier.

The video is shared as "anyone with the link" and a direct-download link is added to the sheet; IFTTT then uploads it (about hourly on its free plan).

## Run it on your own computer

| Interface | Start it with |
| --- | --- |
| Desktop app (Windows) | `run_gui.bat` |
| Web app (any OS) | `pip install -r requirements.txt gradio` then `python gradio_app.py` |
| Command line | `python make_dance_video.py --bpm 124` (see `--help`) |

All three need `ffmpeg` and `ffprobe` on the PATH. They use an NVIDIA GPU (NVDEC + NVENC) by default. Without one, tick CPU in the web app (it picks it automatically) or pass `--cpu-decode --cpu-encode` on the command line.
