#!/usr/bin/env python
# Fake ffmpeg module zodat glassesTools niet crasht op add_to_path()
import types
import sys
import os
import multiprocessing

# Fake 'ffmpeg' module injecteren als het ontbreekt
if "ffmpeg" not in sys.modules:
    fake_ffmpeg = types.SimpleNamespace(add_to_path=lambda: None)
    sys.modules["ffmpeg"] = fake_ffmpeg

print(f"Running script: {os.path.abspath(__file__)}")

# Detecteer base path (PyInstaller vs normaal)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Voeg src/ toe aan sys.path
src_dir = os.path.join(base_path, 'src')
sys.path.insert(0, src_dir)
print(f"Added src directory to path: {src_dir}")
print(f"Python path (eerste 3): {sys.path[:3]}")

# Voeg ffmpeg_bin/bin toe aan PATH zodat ffmpeg.exe gevonden wordt
ffmpeg_bin_dir = os.path.join(base_path, 'ffmpeg_bin', 'bin')
os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")
print(f"ffmpeg path toegevoegd aan PATH: {ffmpeg_bin_dir}")

# Voeg venv_scripts toe aan PATH zodat scripts zoals watchfiles.exe gevonden worden
venv_scripts_dir = os.path.join(base_path, 'venv_scripts')
if os.path.isdir(venv_scripts_dir):
    os.environ["PATH"] = venv_scripts_dir + os.pathsep + os.environ["PATH"]
    print(f"venv_scripts toegevoegd aan PATH: {venv_scripts_dir}")
else:
    print(f"Let op: venv_scripts directory niet gevonden op {venv_scripts_dir}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        import gazeMapper.GUI
        gazeMapper.GUI.run()
    except Exception as e:
        print(f"Fout tijdens uitvoeren van gazeMapper: {e}")
        sys.exit(1)

