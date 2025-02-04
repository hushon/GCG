import cv2
import os
from tqdm import tqdm
import sys
import json
import pandas as pd

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_csv(path):
    file_list = []
    data = pd.read_csv(path)
    columns = data.columns.tolist()
    for index, row in data.iterrows():
        file_list.append({})
        for column in columns:
            file_list[index][column] = row[column]
    return file_list

# def extract_frames(video_path, frame_path, frames=16):
#     breakpoint()

#     # 비디오 파일 열기
#     cap = cv2.VideoCapture(video_path)
#     os.makedirs(frame_path, exist_ok=True)
#     # 비디오 프레임 속도 가져오기
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     # 지정된 시작 프레임으로 이동
#     start_frame = 1
#     # 프레임 추출 간격 계산
#     total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - start_frame
#     frame_interval = max(total_frames // frames, 1)
#     # print(fps, start_frame, total_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT), frame_interval)

#     if total_frames < frames:
#         print(frame_path, f"<{frames} frames!!!!!!!")

#     ret, frame = cap.read()
#     # 프레임 추출 시작
#     frame_count = 0
#     current_frame = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         current_frame += 1
#         # 매 frame_interval 프레임마다 한 프레임 저장
#         if (current_frame - start_frame) % frame_interval == 0 and current_frame >= start_frame:
#             output_file_path = os.path.join(frame_path, f'frame_{frame_count}.jpg')
#             cv2.imwrite(output_file_path, frame)
#             frame_count += 1
#         # 이미 충분한 프레임을 추출한 경우, 루프 조기 종료
#         if frame_count >= frames:
#             break
#     # 비디오 파일 객체 해제
#     cap.release()

#     if frame_count < frames:
#         print(frame_path, f"<{frames} frames!!!!!!!")

import imageio.v3 as iio
def extract_frames(video_path, frame_path, frames=16):

    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    os.makedirs(frame_path, exist_ok=True)
    # 비디오 프레임 속도 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 지정된 시작 프레임으로 이동
    start_frame = 1
    # 프레임 추출 간격 계산
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - start_frame
    frame_interval = max(total_frames // frames, 1)
    # print(fps, start_frame, total_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT), frame_interval)

    if total_frames < frames:
        print(frame_path, f"<{frames} frames!!!!!!!")

    # 프레임 추출 시작
    frame_count = 0
    current_frame = 0
    for frame in iio.imiter(video_path):
        current_frame += 1
        # 매 frame_interval 프레임마다 한 프레임 저장
        if (current_frame - start_frame) % frame_interval == 0 and current_frame >= start_frame:
            output_file_path = os.path.join(frame_path, f'frame_{frame_count}.jpg')
            iio.imwrite(output_file_path, frame)
            frame_count += 1
        # 이미 충분한 프레임을 추출한 경우, 루프 조기 종료
        if frame_count >= frames:
            break

    if frame_count < frames:
        print(frame_path, f"<{frames} frames!!!!!!!")


train_data = load_csv("../nextqa/annotations_mc/train.csv")
val_data = load_csv("../nextqa/annotations_mc/val.csv")
test_data = load_csv("../nextqa/annotations_mc/test.csv")
mapper = load_json('../nextqa/map_vid_vidorID.json')
data = train_data + val_data + test_data

video_ids = []
for item in data:
    video_id = item['video']
    video_ids.append(video_id)
video_ids = list(set(video_ids))

# for video_id in tqdm(video_ids):
#     video_path = f"../nextqa/videos/{mapper[str(video_id)]}.mp4"
#     frame_path = f"../nextqa/frames_32/{video_id}"
#     extract_frames(video_path, frame_path, frames=32)

from joblib import Parallel, delayed
import tqdm.auto as tqdm

def worker_fn(video_id):
    video_path = f"../nextqa/videos/{mapper[str(video_id)]}.mp4"
    frame_path = f"../nextqa/frames_32/{video_id}"
    extract_frames(video_path, frame_path, frames=32)

num_workers = 16
temp_paths = Parallel(num_workers)(delayed(worker_fn)(vid) for vid in tqdm.tqdm(video_ids, total=len(video_ids)))
