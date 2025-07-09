import sys
import os
import multiprocessing

print(f"Running script: {os.path.abspath(__file__)}")

# Current dir en pad naar dummy ffmpeg
current_dir = os.path.dirname(os.path.abspath(__file__))
dummy_path = os.path.join(current_dir, 'ffmpeg')  # Let op: nu heet het 'ffmpeg'

# Voeg dummy ffmpeg toe aan path vóór import
if os.path.isdir(dummy_path):
    sys.path.insert(0, current_dir)  # Voeg de root toe waar 'ffmpeg' instaat

# Voeg src directory toe aan Python path
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

print(f"Added src directory to path: {src_dir}")
print(f"Python path (eerste 3): {sys.path[:3]}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import gazeMapper.GUI
    gazeMapper.GUI.run()
