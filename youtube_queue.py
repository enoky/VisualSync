#!/usr/bin/env python3
"""
Hand a finished video on Google Drive to YouTube, through an IFTTT Applet.

Drive's normal share link for a big file (over ~100 MB) opens a "can't scan for
viruses" page instead of the video, so YouTube can't fetch it. This module:

  1. waits until the video has finished syncing to Google Drive,
  2. shares it as "anyone with the link can view",
  3. builds a direct-download link that skips the virus-scan page, and
  4. adds a row to the Google Sheet "YouTube Queue":
         A = direct link   B = title   C = description

An IFTTT Applet ("New row added to spreadsheet" -> YouTube "Upload video from URL")
watches that sheet and does the upload, so the video never passes through your
phone or computer.

Only works on Google Colab with Drive mounted and the user signed in
(google.colab.auth.authenticate_user()), which the notebook does for you.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

DRIVE_MOUNT = Path("/content/drive")
MY_DRIVE = DRIVE_MOUNT / "MyDrive"
QUEUE_SHEET = "YouTube Queue"
FOLDER_MIME = "application/vnd.google-apps.folder"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"
SYNC_TIMEOUT = 2 * 3600            # how long to wait for Drive to finish uploading the file
POLL_EVERY = 20                    # seconds between checks while it syncs
TITLE_MAX = 100                    # YouTube's limit


class QueueError(RuntimeError):
    """Something the user needs to fix; the message says what."""


# --------------------------------------------------------------------------- #
# Google APIs
# --------------------------------------------------------------------------- #
def _services():
    """Drive v3 and Sheets v4 clients using the Colab sign-in."""
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise QueueError("The Google API libraries aren't installed "
                         "(pip install google-api-python-client google-auth).") from exc
    try:
        creds, _ = google.auth.default()
    except Exception as exc:
        raise QueueError("Not signed in to Google. In the notebook, tick SEND_TO_YOUTUBE in "
                         "cell 2 and run it again (it asks you to sign in), or use cell 5 "
                         "to queue the video.") from exc
    return (build("drive", "v3", credentials=creds, cache_discovery=False),
            build("sheets", "v4", credentials=creds, cache_discovery=False))


def _q(text: str) -> str:
    """Quote a name for a Drive search query."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _list(drive, query: str, fields: str = "id,name,size,createdTime") -> list[dict]:
    resp = drive.files().list(q=query, spaces="drive", pageSize=100,
                              fields=f"files({fields})").execute()
    return resp.get("files", [])


def _folder_id(drive, parts: list[str]) -> str | None:
    """ID of MyDrive/<parts...>, or None if it doesn't exist (yet)."""
    parent = "root"
    for name in parts:
        found = _list(drive, f"name = {_q(name)} and {_q(parent)} in parents and "
                             f"mimeType = '{FOLDER_MIME}' and trashed = false", "id")
        if not found:
            return None
        parent = found[0]["id"]
    return parent


def _drive_parts(path: Path) -> list[str]:
    try:
        rel = path.resolve().relative_to(MY_DRIVE.resolve())
    except (OSError, ValueError):
        raise QueueError(f"{path} isn't in My Drive (/content/drive/MyDrive/…). "
                         "Save the video to a Google Drive folder first.") from None
    return list(rel.parts)


def wait_for_file_id(drive, path: Path) -> Iterator[str]:
    """Yield progress messages until Drive has the whole file; the last item is its ID.

    Colab writes to Drive in the background, so a 3 GB video can take a while to
    appear. Drive reports the full size only once the upload is complete.
    """
    parts = _drive_parts(path)
    size = path.stat().st_size
    deadline = time.monotonic() + SYNC_TIMEOUT
    started = time.monotonic()
    while True:
        folder = _folder_id(drive, parts[:-1])
        if folder:
            files = _list(drive, f"name = {_q(parts[-1])} and {_q(folder)} in parents "
                                 "and trashed = false")
            done = [f for f in files if int(f.get("size", -1)) == size]
            if done:
                yield max(done, key=lambda f: f["createdTime"])["id"]
                return
        if time.monotonic() > deadline:
            raise QueueError(f"{path.name} still hasn't finished syncing to Google Drive "
                             f"after {SYNC_TIMEOUT // 60} minutes. Try again later.")
        mins = int(time.monotonic() - started) // 60
        yield (f"Waiting for {path.name} ({size / 1e9:.1f} GB) to finish syncing to "
               f"Google Drive… ({mins} min so far)")
        time.sleep(POLL_EVERY)


def share_by_link(drive, file_id: str) -> None:
    drive.permissions().create(fileId=file_id, fields="id",
                               body={"type": "anyone", "role": "reader"}).execute()


def unshare(drive, file_id: str) -> None:
    """Undo share_by_link (back to "Restricted")."""
    perms = drive.permissions().list(fileId=file_id, fields="permissions(id,type)").execute()
    for p in perms.get("permissions", []):
        if p["type"] == "anyone":
            drive.permissions().delete(fileId=file_id, permissionId=p["id"]).execute()


def direct_link(file_id: str) -> str:
    """A link that returns the file itself, even when it's too big for a virus scan."""
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def queue_sheet_id(drive) -> str:
    found = _list(drive, f"name = {_q(QUEUE_SHEET)} and mimeType = '{SHEET_MIME}' "
                         "and trashed = false", "id,name")
    if not found:
        raise QueueError(f"No Google Sheet named \"{QUEUE_SHEET}\" in your Drive. Create an "
                         "empty one (the IFTTT Applet watches it).")
    if len(found) > 1:
        raise QueueError(f"There are {len(found)} Google Sheets named \"{QUEUE_SHEET}\". "
                         "Keep only the one the IFTTT Applet watches.")
    return found[0]["id"]


def clean_title(title: str, fallback: str) -> str:
    """YouTube rejects < and > and titles over 100 characters."""
    text = " ".join((title or "").split()) or fallback
    text = text.replace("<", "").replace(">", "").strip()
    return text[:TITLE_MAX].rstrip()


def clean_description(text: str) -> str:
    return (text or "").replace("<", "").replace(">", "").strip()[:5000]


# --------------------------------------------------------------------------- #
# The whole job
# --------------------------------------------------------------------------- #
def queue_video(path: Path | str, title: str = "", description: str = "") -> Iterator[str]:
    """Queue one video for YouTube. Yields progress messages; the last one says it's done.

    Raises QueueError with a readable message if something needs fixing.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise QueueError(f"File not found: {path}")
    drive, sheets = _services()
    sheet_id = queue_sheet_id(drive)       # fail early, before waiting on the sync

    file_id = None
    for item in wait_for_file_id(drive, path):
        if item.startswith("Waiting"):
            yield item
        else:
            file_id = item
    yield "Sharing the video by link…"
    share_by_link(drive, file_id)

    row = [direct_link(file_id), clean_title(title, path.stem), clean_description(description)]
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id, range="A:C", valueInputOption="RAW",
        insertDataOption="INSERT_ROWS", body={"values": [row]}).execute()
    yield (f"Queued \"{row[1]}\" for YouTube. IFTTT checks the sheet about once an hour on "
           "the free plan; the video then appears in YouTube Studio as Private.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Queue a Drive video for YouTube via IFTTT")
    ap.add_argument("video", help="path under /content/drive/MyDrive")
    ap.add_argument("--title", default="", help="YouTube title (default: the file name)")
    ap.add_argument("--description", default="")
    args = ap.parse_args()
    try:
        for msg in queue_video(args.video, args.title, args.description):
            print(msg, flush=True)
    except QueueError as exc:
        raise SystemExit(f"Not queued: {exc}")


if __name__ == "__main__":
    main()
