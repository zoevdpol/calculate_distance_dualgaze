from gazeMapper.mapper import Mapper


import cv2
import pandas as pd
import json

# === INPUTBESTANDEN ===
video_path = 'video.mp4'
gaze_path = 'gaze.csv'
marker_config_path = 'markers.json'
output_video_path = 'annotated_output.mp4'

# === LAAD DATA ===
gaze_df = pd.read_csv(gaze_path)
gaze_data = gaze_df.to_dict(orient='records')

with open(marker_config_path, 'r') as f:
    marker_config = json.load(f)

# === MAPPEN VAN GAZE ===
mapper = Mapper(marker_config, video_path, gaze_data)
mapped_gaze = mapper.map_gaze()

# === VIDEO MET GAZE ANNOTATIE MAKEN ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_idx / fps

    # Teken gaze punten die binnen dit frame vallen
    for gaze in mapped_gaze:
        if abs(gaze['timestamp'] - current_time) < (1 / fps):
            if gaze['valid']:
                x, y = int(gaze['x_mapped']), int(gaze['y_mapped'])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # groene stip

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Video met gaze-output opgeslagen.")
