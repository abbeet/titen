# museum/utils.py

import cv2
import os
import re
import time
import torch
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point
from collections import defaultdict
from django.utils import timezone
from .models import ROI, DetectionResults

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU
    return reader

def enhance_frame(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    if frame.ndim == 3:  # If the frame is colored
        yuv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        yuv_img[:, :, 0] = clahe.apply(yuv_img[:, :, 0])
        enhanced_frame = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    elif frame.ndim == 2:  # If the frame is grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_frame = clahe.apply(frame)
    else:
        raise ValueError("Unsupported frame dimension")
    return enhanced_frame

def extract_timestamp(frame, roi):
    x, y, w, h = roi
    cropped_frame = frame[y:y+h, x:x+w]
    
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    reader = get_ocr_reader()
    result = reader.readtext(thresh_frame, detail=0)
    
    ocr_text = ''.join(result)
    timestamp = extract_components(ocr_text)
    
    return timestamp.strip()

def extract_components(ocr_text):
    digits = re.findall(r'\d', ocr_text)
    if len(digits) >= 14:  # Ensure there are enough digits for YYYYMMDDHHMMSS
        year = ''.join(digits[:4])
        month = ''.join(digits[4:6])
        day = ''.join(digits[6:8])
        hour = ''.join(digits[8:10])
        minute = ''.join(digits[10:12])
        second = ''.join(digits[12:14])
        return f"{year}-{month}-{day} {hour}:{minute}:{second}"
    return ""

def format_timestamp(text):
    datetime_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    match = datetime_pattern.search(text)
    if match:
        try:
            timestamp = match.group(0)
            formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            return formatted_timestamp
        except ValueError:
            return None
    return None

def load_rois(cctv_name_id):
    regions = []
    rois = ROI.objects.filter(cctv_name_id=cctv_name_id)

    for roi in rois:
        coords = [tuple(map(float, point.split(','))) for point in roi.coordinates.split(';')]
        if len(coords) < 3:
            raise ValueError(f"Invalid ROI coordinates for {roi.roi_name}: {roi.coordinates}")
        regions.append({
            "name": roi.roi_name,
            "polygon": Polygon(coords),
            "counts": 0,
            "total_counts": 0,
            "long_stay_counts": 0,
            "individual_stay_times": defaultdict(int),
            "individual_start_times": {},
            "tracked_ids": set(),
            "region_color": (255, 42, 4),
            "text_color": (255, 165, 0),
        })
    return regions

def reset_all(counting_regions):
    for region in counting_regions:
        region["counts"] = 0
        region["total_counts"] = 0
        region["long_stay_counts"] = 0
        region["individual_stay_times"].clear()
        region["individual_start_times"].clear()
        region["tracked_ids"].clear()

def process_video(roi_source_id, video_source, is_feed_url, tracker="botsort.yaml"):
    interval_start_time, interval_end_time = None, None
    interval_data = defaultdict(lambda: {"visitor_passing_count": 0, "visitor_interested_count": 0})
    counting_regions = load_rois(roi_source_id)

    model = YOLO(f"{settings.BASE_DIR}/museum/models/best9.pt")

    if settings.USE_GPU and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    video_capture = cv2.VideoCapture(video_source) if is_feed_url else cv2.VideoCapture(str(video_source))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    target_fps = 5
    frame_skip_interval = int(video_fps / target_fps)
    frame_count = 0
    processed_frames = 0
    timestamps = []

    reset_all(counting_regions)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_skip_interval != 0:
            frame_count += 1
            continue

        frame_count += 1
        processed_frames += 1
        frame = enhance_frame(frame)

        conf = 0.2
        iou = 0.5
        results = model.track(frame, persist=True, conf=conf, iou=iou, tracker=tracker)

        for region in counting_regions:
            region["counts"] = 0

        if results[0].boxes.xyxy.numel() != 0:
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    if cls == 0:
                        bbox_foot = (box[0] + box[2]) / 2, box[3]

                        for region in counting_regions:
                            if region["polygon"].contains(Point(bbox_foot)):
                                region["counts"] += 1
                                if track_id not in region["tracked_ids"]:
                                    region["total_counts"] += 1
                                    region["tracked_ids"].add(track_id)
                                    region["individual_start_times"][track_id] = time.time()

                                duration = time.time() - region["individual_start_times"][track_id]
                                region["individual_stay_times"][track_id] = duration
                                if duration >= 10:
                                    region["long_stay_counts"] = len([t for t in region["individual_stay_times"].values() if t >= 10])

                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        for region in counting_regions:
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region["region_color"], thickness=2)
            label = f"{region['name']} Count: {region['counts']}, Total: {region['total_counts']}, Long Stay: {region['long_stay_counts']}"
            cv2.putText(frame, label, (10, 30 + 30 * counting_regions.index(region)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region["text_color"], 2)

        timestamp = extract_timestamp(frame, (929, 11, 330, 50))
        formatted_timestamp = format_timestamp(timestamp)
        if formatted_timestamp:
            timestamps.append(formatted_timestamp)
            current_time = timezone.make_aware(datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S"))

            if not interval_start_time:
                interval_start_time = current_time
                interval_end_time = interval_start_time + timedelta(minutes=10)

            if current_time > interval_end_time:
                room = None
                if not is_feed_url:
                    footage = CCTVFootages.objects.get(id=roi_source_id)
                    room = footage.cctv_name.room_name

                for region in counting_regions:
                    try:
                        roi = ROI.objects.get(roi_name=region['name'], cctv_name_id=cctv_name_id)
                        DetectionResults.objects.create(
                            roi_name=roi,
                            room=room,
                            date=interval_start_time.date(),
                            time_start=interval_start_time.time(),
                            time_end=interval_end_time.time(),
                            visitor_passing_count=interval_data[region["name"]]["visitor_passing_count"],
                            visitor_interested_count=interval_data[region["name"]]["visitor_interested_count"]
                        )
                    except ROI.DoesNotExist:
                        print(f"ROI not found for roi_name: {region['name']} and cctv_name_id: {cctv_name_id}")

                interval_start_time = current_time
                interval_end_time = interval_start_time + timedelta(minutes=10)
                reset_all(counting_regions)

            for region in counting_regions:
                interval_data[region["name"]]["visitor_passing_count"] = region["total_counts"]
                interval_data[region["name"]]["visitor_interested_count"] = region["long_stay_counts"]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_capture.release()
    cv2.destroyAllWindows()

    return counting_regions, timestamps
