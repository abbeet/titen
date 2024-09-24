# museum/views.py

from django.conf import settings
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.core.serializers.json import DjangoJSONEncoder
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from .forms import MuseumForm, RoomForm, CCTVFeedForm, CCTVFootagesForm
from .models import Room, CCTVFeed, CCTVFootages, ROI, DetectionResults

import cv2
import easyocr
import os
import pytz
import re
import time
import torch
import numpy as np

from collections import defaultdict
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point
from ultralytics import YOLO

# Global variables
rois_points = []  # List to store ROIs points
current_roi = []  # Temporary storage for the current ROI being drawn
saved_rois = []  # List to store saved ROIs from the file
video_paused = False  # Flag to control video pause for ROI drawing
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU
local_tz = pytz.timezone('Asia/Bangkok')
interval_start_time = None
interval_end_time = None
interval_data = defaultdict(lambda: {"visitor_passing_count": 0, "visitor_interested_count": 0})
counting_regions = []

def welcome(request):
    return render(request, 'museum/welcome.html')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'museum/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'museum/login.html', {'form': form})

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
    """Extract timestamp from a given frame using EasyOCR."""
    x, y, w, h = roi
    cropped_frame = frame[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform OCR using EasyOCR
    result = reader.readtext(thresh_frame, detail=0)
    print("Raw OCR result:", result)  # Debugging: Print the raw OCR result
    
    # Combine OCR results into a single string
    ocr_text = ''.join(result)
    
    # Extract timestamp components
    timestamp = extract_components(ocr_text)
    print("Extracted timestamp:", timestamp)  # Debugging: Print the extracted timestamp
    
    return timestamp.strip()

def extract_components(ocr_text):
    """Extract components of the timestamp from OCR text."""
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
    """Format the extracted text as a timestamp."""
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

def reset_all():
    """Reset global variables and all interval data."""
    global interval_start_time, interval_end_time, interval_data, counting_regions
    print("Resetting all data")  # Debug print
    interval_start_time = None
    interval_end_time = None
    interval_data = defaultdict(lambda: {"visitor_passing_count": 0, "visitor_interested_count": 0})
    for region in counting_regions:
        region["counts"] = 0
        region["total_counts"] = 0
        region["long_stay_counts"] = 0
        region["individual_stay_times"].clear()
        region["individual_start_times"].clear()
        region["tracked_ids"].clear()

def process_video(roi_source_id, video_source, is_feed_url, tracker="botsort.yaml"):
    global interval_start_time, interval_end_time, interval_data, counting_regions

    print(f"Starting video processing for source ID: {roi_source_id}, is_feed_url: {is_feed_url}")

    # Load the YOLO model
    model = YOLO(f"{settings.BASE_DIR}/museum/models/best9.pt")

    # Move the model to GPU if available and configured
    if settings.USE_GPU and torch.cuda.is_available():
        model.to("cuda")
        print("Using GPU for detection.")
    else:
        model.to("cpu")
        print("Using CPU for detection.")

    video_capture = cv2.VideoCapture(video_source) if is_feed_url else cv2.VideoCapture(str(video_source))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Set the target FPS to 5
    target_fps = 5
    frame_skip_interval = int(video_fps / target_fps)
    print(f"Video FPS: {video_fps}, Processing every {frame_skip_interval} frame(s) to achieve {target_fps} FPS.")

    # Determine room and cctv_name_id based on the source
    if is_feed_url:
        cctv_name_id = roi_source_id
        feed = CCTVFeed.objects.get(id=cctv_name_id)
        room = feed.room_name  # Assign the room from the feed
    else:
        footage = CCTVFootages.objects.get(id=roi_source_id)
        cctv_name_id = footage.cctv_name.id
        room = footage.cctv_name.room_name  # Assign the room from the footage

    counting_regions = load_rois(cctv_name_id)
    print(f"Loaded {len(counting_regions)} ROIs for cctv_name_id: {cctv_name_id}")

    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    timestamps = []

    reset_all()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Skip frames to achieve the target FPS
        if frame_count % frame_skip_interval != 0:
            frame_count += 1
            continue

        frame_count += 1
        processed_frames += 1
        frame = enhance_frame(frame)

        # Run YOLOv9 tracking on the frame, use the specified tracker
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

        if not formatted_timestamp:
            current_time_utc = timezone.now()
            current_time_local = current_time_utc.astimezone(local_tz)
            formatted_timestamp = current_time_local.strftime("%Y-%m-%d %H:%M:%S")
            print(f"Using current time as timestamp (GMT +7): {formatted_timestamp}")
        else:
            current_time_local = timezone.make_aware(datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S"), timezone=local_tz)
        
        timestamps.append(formatted_timestamp)
        current_time = timezone.make_aware(datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S"))

        if not interval_start_time:
            interval_start_time = timezone.make_aware(datetime.strptime(formatted_timestamp, "%Y-%m-%d %H:%M:%S"), timezone=local_tz)
            interval_end_time = interval_start_time + timedelta(minutes=10)

        if current_time > interval_end_time:
            room = None
            if not is_feed_url:
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
            reset_all()

        for region in counting_regions:
            interval_data[region["name"]]["visitor_passing_count"] = region["total_counts"]
            interval_data[region["name"]]["visitor_interested_count"] = region["long_stay_counts"]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_capture.release()
    cv2.destroyAllWindows()

    if interval_start_time:
        for region in counting_regions:
            try:
                roi = ROI.objects.get(roi_name=region['name'], cctv_name_id=cctv_name_id)
                DetectionResults.objects.create(
                    roi_name=roi,
                    room=room,  # Now the room will always be assigned correctly
                    date=interval_start_time.date(),
                    time_start=interval_start_time.time(),
                    time_end=interval_end_time.time(),
                    visitor_passing_count=interval_data[region["name"]]["visitor_passing_count"],
                    visitor_interested_count=interval_data[region["name"]]["visitor_interested_count"]
                )
            except ROI.DoesNotExist:
                print(f"ROI not found for roi_name: {region['name']} and cctv_name_id: {cctv_name_id}")

    return counting_regions, timestamps

@login_required
def logout_view(request):
    logout(request)
    return redirect('welcome')

@login_required
def adjust_brightness_clahe(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Adjusts the frame's brightness using CLAHE."""
    if frame.ndim == 3:  # Colored frame (BGR)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        adjusted_frame = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2BGR)
    elif frame.ndim == 2:  # Grayscale frame
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        adjusted_frame = clahe.apply(frame)
    else:
        raise ValueError("Unsupported frame dimension")
    return adjusted_frame

@login_required
def save_rois_to_db(rois_points, feed):
    for roi in rois_points:
        coordinates = ';'.join([f"{x},{y}" for x, y in roi])
        ROI.objects.create(cctv_name=feed, roi_name="ROI", coordinates=coordinates)

@login_required
def dashboard(request):
    return render(request, 'museum/dashboard.html')

@login_required
def add_museum(request):
    if request.method == 'POST':
        form = MuseumForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = MuseumForm()
    return render(request, 'museum/add_museum.html', {'form': form})

@login_required
def add_rooms(request):
    if request.method == 'POST':
        form = RoomForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = RoomForm()
    return render(request, 'museum/add_rooms.html', {'form': form})

@login_required
def add_cctv_feed(request):
    if request.method == 'POST':
        form = CCTVFeedForm(request.POST, request.FILES)
        if form.is_valid():
            cctv_feed = form.save()

            # Capture the first frame of the CCTV feed
            cap = cv2.VideoCapture(cctv_feed.feed_url)
            ret, frame = cap.read()
            if ret:
                frame_dir = os.path.join(settings.MEDIA_ROOT, 'first_frames')
                os.makedirs(frame_dir, exist_ok=True)

                frame_filename = f'{cctv_feed.id}.jpg'
                frame_path = os.path.join(frame_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                cctv_feed.first_frame_image = f'first_frames/{frame_filename}'
                cctv_feed.save()

            cap.release()
            return redirect('dashboard')
    else:
        form = CCTVFeedForm()
    return render(request, 'museum/add_cctv_feed.html', {'form': form})

@login_required
def add_cctv_footages(request):
    if request.method == 'POST':
        form = CCTVFootagesForm(request.POST, request.FILES)
        if form.is_valid():
            cctv_name = form.cleaned_data['cctv_name']
            video_files = form.cleaned_data['video_files']
            for video_file in video_files:
                CCTVFootages.objects.create(cctv_name=cctv_name, video_file=video_file)
            return redirect('dashboard')
    else:
        form = CCTVFootagesForm()
    return render(request, 'museum/add_cctv_footages.html', {'form': form})

@login_required
def list_cctv(request):
    feeds = CCTVFeed.objects.all()
    return render(request, 'museum/list_cctv.html', {'feeds': feeds})

@login_required
def draw_roi(request, feed_id):
    feed = get_object_or_404(CCTVFeed, id=feed_id)

    if request.method == 'POST':
        coordinates = request.POST.get('coordinates')
        if coordinates:
            rois = coordinates.split('|')
            for roi in rois:
                label, coordinates_str = roi.split(':')
                ROI.objects.create(cctv_name=feed, roi_name=label, coordinates=coordinates_str)
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No coordinates provided'}, status=400)

    frame_path = feed.first_frame_image.url if feed.first_frame_image else ''
    return render(request, 'museum/draw_roi.html', {'feed': feed, 'frame_path': frame_path})

@login_required
def process_video_feed(request, feed_id):
    feed = CCTVFeed.objects.get(id=feed_id)
    rtsp_url = feed.feed_url
    
    cap = cv2.VideoCapture(rtsp_url)
    
    def gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Process frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@login_required
def run_detection(request):
    if request.method == 'POST':
        feed_id = request.POST.get('feed_id')
        footage_id = request.POST.get('footage_id')

        if feed_id:
            feed = CCTVFeed.objects.get(id=feed_id)
            video_source = feed.feed_url
            is_feed_url = True
            roi_source_id = feed.id  # Correctly assign roi_source_id for feed
            print(f"Running detection for feed ID: {feed_id}, roi_source_id set to: {roi_source_id}")
        elif footage_id:
            footage = CCTVFootages.objects.get(id=footage_id)
            video_source = os.path.join(settings.MEDIA_ROOT, footage.video_file.name)
            is_feed_url = False
            roi_source_id = footage.cctv_name.id  # Correctly assign roi_source_id for footage
            print(f"Running detection for footage ID: {footage_id}, associated feed ID: {footage.cctv_name.id}, roi_source_id set to: {roi_source_id}")
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid request'})

        results, timestamps = process_video(roi_source_id, video_source, is_feed_url, tracker="botsort.yaml")
        return StreamingHttpResponse(results, content_type='multipart/x-mixed-replace; boundary=frame')

    rooms = Room.objects.all()
    return render(request, 'museum/run_detection.html', {'rooms': rooms})

@csrf_exempt
@login_required
def stream_video(request, feed_id=None, footage_id=None):
    if feed_id:
        feed = get_object_or_404(CCTVFeed, id=feed_id)
        video_source = feed.feed_url
        is_feed_url = True
        roi_source_id = feed_id  # Correctly assign roi_source_id for feed
        print(f"Streaming video for feed ID: {feed_id}, roi_source_id set to: {roi_source_id}")
    elif footage_id:
        footage = get_object_or_404(CCTVFootages, id=footage_id)
        video_source = os.path.join(settings.MEDIA_ROOT, footage.video_file.name)
        is_feed_url = False
        roi_source_id = footage.id  # Correctly assign roi_source_id for footage
        print(f"Streaming video for footage ID: {footage_id}, associated feed ID: {footage.cctv_name.id}, roi_source_id set to: {roi_source_id}")
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request'})

    response = StreamingHttpResponse(process_video(roi_source_id, video_source, is_feed_url),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    return response

@login_required
def get_feeds_and_footages(request):
    room_id = request.GET.get('room_id')
    if room_id:
        feeds = CCTVFeed.objects.filter(room_name_id=room_id).values('id', 'cctv_name')
        footages = CCTVFootages.objects.filter(cctv_name__room_name_id=room_id).values('id', 'video_file')
        return JsonResponse({'feeds': list(feeds), 'footages': list(footages)})
    return JsonResponse({'feeds': [], 'footages': []})

@login_required
def view_results(request):
    # Mengambil semua hasil deteksi dari database
    results = DetectionResults.objects.select_related('roi_name', 'room').all()

    # Menyusun data untuk grafik dan tabel
    data = []
    roi_names = []
    total_counts = []
    long_stay_counts = []
    room_names = []
    museum_names = []
    time_starts = []
    time_ends = []

    for result in results:
        roi_name = result.roi_name.roi_name
        room_name = result.room.room_name
        museum_name = result.room.museum_name.museum_name
        time_start = result.time_start.strftime('%H:%M:%S')
        time_end = result.time_end.strftime('%H:%M:%S')

        if roi_name not in roi_names:
            roi_names.append(roi_name)
            total_counts.append(result.visitor_passing_count)
            long_stay_counts.append(result.visitor_interested_count)
            room_names.append(room_name)
            museum_names.append(museum_name)
            time_starts.append(time_start)
            time_ends.append(time_end)
        else:
            index = roi_names.index(roi_name)
            total_counts[index] += result.visitor_passing_count
            long_stay_counts[index] += result.visitor_interested_count

        # Data untuk tabel
        data.append({
            'museum_name': museum_name,
            'room_name': room_name,
            'roi_name': roi_name,
            'date': result.date,
            'time_start': time_start,
            'time_end': time_end,
            'visitor_passing_count': result.visitor_passing_count,
            'visitor_interested_count': result.visitor_interested_count,
        })

    context = {
        'results': data,
        'roi_names': roi_names,
        'total_counts': total_counts,
        'long_stay_counts': long_stay_counts,
        'room_names': room_names,
        'museum_names': museum_names,
        'time_starts': time_starts,
        'time_ends': time_ends,
    }

    return render(request, 'museum/view_results.html', context)