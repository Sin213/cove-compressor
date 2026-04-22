# Cove Compressor

A clean, dark-themed GUI for **offline batch image and video compression** —
no cloud, no API keys, no accounts. Built with PySide6, Pillow, and ffmpeg.

One codebase, one repository, native builds for both platforms: a Windows
installer + portable exe, and a Linux AppImage + .deb. Every `v*` tag cuts
all four artifacts via GitHub Actions.

![Python](https://img.shields.io/badge/python-3.10%2B-orange?style=flat-square&logo=python)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-informational?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Features

### Images
- Batch compress JPEG, PNG, WebP, and AVIF
- Three quality presets — **Light**, **Balanced**, **Aggressive**
- Force output format — Keep original, JPEG, WebP, or AVIF
- Optional resize cap (longest edge: 1280 / 1920 / 2560 / 4000 px)
- EXIF orientation respected automatically
- Progressive JPEG + WebP method 6 for smaller output
- Skips files where compression would *increase* the size
- Parallel processing (up to 8 worker threads)

### Videos
- **Target file size** — 2-pass encode to hit an exact MB target
- **Target reduction** — encode to a % of the original size
- **Quality preset** — CRF-based (Web Small / Balanced / Archive Light) *(default)*
- Codec choice: **H.265 (x265)** or **H.264 (x264)**
- Resolution cap: Original / 1080p / 720p / 480p
- Audio bitrate: 128 / 192 / 320 kbps
- Skips if output would be larger than the input
- Live progress bar and log

### General
- Drag & drop files or folders onto the window
- Separate tabs for Images and Videos
- Outputs to `~/Downloads/cove-compressed` (Linux/macOS) or
  `%USERPROFILE%\Downloads\cove-compressed` (Windows) by default
- Fully dark-themed UI (Fusion style + custom palette on Windows)
- Cancel button for in-progress batches

---

## Install a prebuilt release

Head to the [Releases page](https://github.com/Sin213/cove-compressor/releases)
and grab the artifact for your OS:

| OS      | Artifact                                      | Notes                                         |
| ------- | --------------------------------------------- | --------------------------------------------- |
| Windows | `cove-compressor-<version>-Setup.exe`         | Inno Setup installer (Start Menu + Desktop)   |
| Windows | `cove-compressor-<version>-Portable.exe`      | Single-file, no install                       |
| Linux   | `Cove-Compressor-<version>-x86_64.AppImage`   | `chmod +x` and run                            |
| Linux   | `cove-compressor_<version>_amd64.deb`         | `sudo apt install ./cove-compressor_*.deb`    |

`ffmpeg` and `ffprobe` are **bundled inside every artifact** — no separate
install needed on either platform.

> **Windows SmartScreen** may warn on first launch because the exe isn't
> signed. Click **More info → Run anyway**.

---

## Running from source (Linux)

Python 3.10+. On Arch:

```bash
sudo pacman -S python python-pillow pyside6 ffmpeg
python cove_compressor.py
```

On Debian / Ubuntu:

```bash
sudo apt install python3 python3-pil python3-pyside6.qtwidgets ffmpeg
python3 cove_compressor.py
```

Or via a venv (any distro):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python cove_compressor.py
```

---

## Running from source (Windows)

Python 3.10+ from [python.org](https://www.python.org/downloads/) (tick
**"Add python.exe to PATH"** during install).

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# ffmpeg, either via winget…
winget install Gyan.FFmpeg
# …or drop ffmpeg.exe + ffprobe.exe next to cove_compressor.py

.venv\Scripts\python cove_compressor.py
```

---

## Building release artifacts yourself

PyInstaller can't cross-compile, so each platform has its own script. Both
download ffmpeg automatically.

### Linux — AppImage + .deb

```bash
bash scripts/build-release.sh
# Output in release/:
#   Cove-Compressor-1.0.0-x86_64.AppImage
#   cove-compressor_1.0.0_amd64.deb
```

Override the version with `VERSION=1.2.0 bash scripts/build-release.sh`.

### Windows — Setup.exe + Portable.exe

Requires [Inno Setup 6](https://jrsoftware.org/isdl.php) (pre-installed on
GitHub Actions' `windows-latest`).

```powershell
.\build.ps1 -Version 1.0.0
# Output in release\:
#   cove-compressor-1.0.0-Setup.exe
#   cove-compressor-1.0.0-Portable.exe
```

### Automated release via GitHub Actions

Push a tag matching `v*` (e.g. `v1.0.0`) and `.github/workflows/release.yml`
runs the Linux + Windows jobs in parallel and attaches all four artifacts to
the GitHub Release created for the tag.

---

## Defaults

| Tab    | Setting           | Default              |
|--------|-------------------|----------------------|
| Images | Preset            | Balanced             |
| Images | Output format     | Keep original        |
| Images | Resize cap        | No cap               |
| Videos | Method            | Quality preset       |
| Videos | Preset            | Balanced             |
| Videos | Codec             | H.265 (x265)         |
| Videos | Resolution cap    | Original             |
| Videos | Audio bitrate     | 192 kbps             |

---

## Licensing

- Cove Compressor is **MIT** — see `LICENSE`.
- The bundled `ffmpeg` / `ffprobe` binaries are the **gyan.dev
  release-essentials** (Windows) and **johnvansickle.com static** (Linux)
  builds, both **GPLv3**. Cove Compressor shells out to those binaries
  rather than linking, so the app's MIT licensing stands. If you
  redistribute release artifacts, comply with the ffmpeg GPL terms — most
  commonly by keeping `FFMPEG-LICENSE.txt` alongside the binary and pointing
  recipients at [ffmpeg.org](https://ffmpeg.org/) for sources.
