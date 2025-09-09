import cv2
import os
import time
import pickle
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime, date
from collections import deque

# MODE: 0 = Enroll (capture/register faces), 1 = Attendance (mark attendance only)
MODE = 0

# Configuration
KNOWN_FACES_DIR = "known_faces"
EXCEL_PATH = "students.xlsx"  # master students registry (Name, Class, SUCNumber,Time)
ATTENDANCE_DIR = "attendance"  # per-day attendance Excel directory
ENCODINGS_CACHE = "encodings_cache.pkl"
FACE_DETECTION_MODEL = "hog"
UNKNOWN_PROMPT_COOLDOWN_SEC = 8.0
RECOGNITION_TOLERANCE = 0.42
RECOGNITION_MARGIN = 0.10
FALLBACK_TOLERANCE = 0.50  # fallback acceptance if KNN+smoothing fails
DOWNSCALE = 0.4
PROCESS_EVERY_N_FRAMES = 5
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360
MULTIFRAME_CONFIRM = 3
TRACK_TIMEOUT_SEC = 1.2
REVALIDATE_EVERY_N_FRAMES = 9999
AREA_CHANGE_MAX_RATIO = 2.0
ROI_MARGIN = 0.16

# Adaptive performance controls
TARGET_FPS = 40.0
MIN_DOWNSCALE = 0.4
MAX_DOWNSCALE = 0.6
MIN_SKIP = 4
MAX_SKIP = 9
FPS_SMOOTHING = 0.15
ADAPT_INTERVAL_SEC = 0.75
DETECTION_INTERVAL_SEC = 1.0

# KNN recognition + smoothing
KNN_K = 3
SMOOTH_WINDOW = 8
CONFIDENCE_MIN = 0.55
label_history = deque(maxlen=SMOOTH_WINDOW)

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# Helper to parse tracked_label like "Name (85%)" back to label key
def _parse_label_from_tracked(text):
    if not text:
        return ""
    # strip confidence part
    if "(" in text:
        base = text.split("(")[0].strip()
    else:
        base = text.strip()
    # keep potential __roll suffix if present
    return base

# Update cache to disk
def _write_encodings_cache(signature, encodings, names):
    try:
        with open(ENCODINGS_CACHE, "wb") as f:
            pickle.dump({
                "signature": signature,
                "encodings": encodings,
                "names": names,
            }, f)
    except Exception:
        pass

# Add extra sample for a label from current ROI, update memory and cache
def add_sample_for_label(frame_bgr, bbox, label, known_encodings, known_names):
    left, top, width, height = bbox
    right = left + width
    bottom = top + height
    h, w = frame_bgr.shape[:2]
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)
    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0:
        return False
    # Save image
    i = 1
    base = label
    save_name = f"{base}.jpg"
    save_path = os.path.join(KNOWN_FACES_DIR, save_name)
    while os.path.exists(save_path):
        save_name = f"{base}_{i}.jpg"
        save_path = os.path.join(KNOWN_FACES_DIR, save_name)
        i += 1
    cv2.imwrite(save_path, crop)
    # Encode and add
    try:
        img = face_recognition.load_image_file(save_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(base)
            signature = _known_faces_state_signature(KNOWN_FACES_DIR)
            _write_encodings_cache(signature, known_encodings, known_names)
            return True
    except Exception:
        pass
    return False

# Simple KNN-style classification
def classify_face_knn(face_encoding, known_encodings, known_names):
    if not known_encodings:
        return "Unknown", 0.0
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    if len(distances) == 0:
        return "Unknown", 0.0
    k = min(KNN_K, len(distances))
    nn_idx = np.argpartition(distances, k - 1)[:k]
    label_to_dists = {}
    for idx in nn_idx:
        label = known_names[idx]
        d = float(distances[idx])
        label_to_dists.setdefault(label, []).append(d)
    label_avgs = {lab: float(np.mean(ds)) for lab, ds in label_to_dists.items()}
    ordered = sorted(label_avgs.items(), key=lambda x: x[1])
    best_label, best_dist = ordered[0]
    second_dist = ordered[1][1] if len(ordered) > 1 else 1.0
    if (best_dist < RECOGNITION_TOLERANCE) and ((second_dist - best_dist) >= RECOGNITION_MARGIN):
        conf = max(0.0, min(1.0, 1.0 - (best_dist / 0.6)))
        return best_label, conf
    return "Unknown", 0.0

# Smoothing
def smooth_label(new_label, new_conf):
    label_history.append((new_label, new_conf))
    counts = {}
    conf_sum = {}
    for lab, c in label_history:
        if lab == "Unknown":
            continue
        counts[lab] = counts.get(lab, 0) + 1
        conf_sum[lab] = conf_sum.get(lab, 0.0) + c
    if not counts:
        return "Unknown", 0.0
    best_lab = max(counts.keys(), key=lambda l: (counts[l], conf_sum[l] / max(1, counts[l])))
    avg_conf = conf_sum[best_lab] / max(1, counts[best_lab])
    if counts[best_lab] >= (SMOOTH_WINDOW // 2):
        return best_lab, avg_conf
    return "Unknown", 0.0

# Fallback: best-only distance accept
def fallback_best_only(face_encoding, known_encodings, known_names):
    if not known_encodings:
        return "Unknown", 0.0
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    if len(distances) == 0:
        return "Unknown", 0.0
    idx = int(np.argmin(distances))
    d = float(distances[idx])
    if d < FALLBACK_TOLERANCE:
        conf = max(0.0, min(1.0, 1.0 - (d / 0.6)))
        return known_names[idx], conf
    return "Unknown", 0.0

# High-quality recheck on full-res ROI
def recheck_fullres_roi(frame_bgr, box, known_encodings, known_names):
    top, right, bottom, left = box
    h, w = frame_bgr.shape[:2]
    top = max(0, top); left = max(0, left)
    bottom = min(h, bottom); right = min(w, right)
    roi = frame_bgr[top:bottom, left:right]
    if roi.size == 0:
        return "Unknown", 0.0
    # Apply CLAHE on L channel in LAB space to normalize lighting
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    roi_eq = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(roi_eq, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if not encs:
        return "Unknown", 0.0
    enc = encs[0]
    lab, conf = classify_face_knn(enc, known_encodings, known_names)
    if lab == "Unknown":
        lab, conf = fallback_best_only(enc, known_encodings, known_names)
    return lab, conf


def _known_faces_state_signature(known_dir):
    signature = []
    for filename in sorted(os.listdir(known_dir)):
        path = os.path.join(known_dir, filename)
        if os.path.isfile(path):
            try:
                stat = os.stat(path)
                signature.append((filename, int(stat.st_mtime)))
            except OSError:
                continue
    return signature


def load_known_faces_from_cache_or_build(known_dir, cache_path):
    signature = _known_faces_state_signature(known_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("signature") == signature:
                return cached.get("encodings", []), cached.get("names", [])
        except Exception:
            pass

    known_encodings = []
    known_names = []
    for filename in os.listdir(known_dir):
        file_path = os.path.join(known_dir, filename)
        if not os.path.isfile(file_path):
            continue
        name, ext = os.path.splitext(filename)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
        except Exception:
            continue

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "signature": signature,
                "encodings": known_encodings,
                "names": known_names,
            }, f)
    except Exception:
        pass

    return known_encodings, known_names


def save_registration_to_excel(name, roll_number, class_name, suc_number, excel_path):
    row = {
        "Name": name,
        "RollNumber": roll_number,
        "Class": class_name,
        "SUCNumber": suc_number,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
        except Exception:
            df = pd.DataFrame(columns=row.keys())
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_excel(excel_path, index=False)


# Lookup helpers
def _extract_name_roll(label):
    if "__" in label:
        name, roll = label.split("__", 1)
    else:
        name, roll = label, ""
    return name, roll

def _lookup_suc_from_students(name, roll, excel_path):
    if not os.path.exists(excel_path):
        return ""
    try:
        df = pd.read_excel(excel_path)
        mask = (df["Name"].astype(str) == name) & (df["RollNumber"].astype(str) == roll)
        matches = df[mask]
        if not matches.empty:
            return str(matches.iloc[0].get("SUCNumber", ""))
    except Exception:
        pass
    return ""

def _lookup_class_from_students(name, roll, excel_path):
    if not os.path.exists(excel_path):
        return ""
    try:
        df = pd.read_excel(excel_path)
        mask = (df["Name"].astype(str) == name) & (df["RollNumber"].astype(str) == roll)
        matches = df[mask]
        if not matches.empty:
            return str(matches.iloc[0].get("Class", ""))
    except Exception:
        pass
    return ""

# Attendance buffering (per day file with SNo, SUC, Name, Class)
attendance_buffer = []
next_sno_today = 1
last_flush_time = time.time()
BUFFER_MIN_ROWS = 5
BUFFER_MAX_WAIT_SEC = 10.0

def _day_path():
    return os.path.join(ATTENDANCE_DIR, f"{date.today().strftime('%Y-%m-%d')}.xlsx")

def init_attendance_state():
    global next_sno_today
    path = _day_path()
    if not os.path.exists(path):
        next_sno_today = 1
        return
    try:
        df = pd.read_excel(path)
        if "SNo" in df.columns and not df.empty:
            next_sno_today = int(df["SNo"].max()) + 1
        else:
            next_sno_today = 1
    except Exception:
        next_sno_today = 1


def flush_attendance_buffer():
    global attendance_buffer, last_flush_time
    if not attendance_buffer:
        return
    try:
        path = _day_path()
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
            except Exception:
                df = pd.DataFrame(columns=["SNo", "SUC", "Name", "Class"])
        else:
            df = pd.DataFrame(columns=["SNo", "SUC", "Name", "Class"])
        df = pd.concat([df, pd.DataFrame(attendance_buffer)], ignore_index=True)
        df.to_excel(path, index=False)
        attendance_buffer = []
        last_flush_time = time.time()
    except Exception:
        pass


def save_attendance_buffered(name_label):
    global next_sno_today
    name, roll = _extract_name_roll(name_label)
    suc = _lookup_suc_from_students(name, roll, EXCEL_PATH)
    cls = _lookup_class_from_students(name, roll, EXCEL_PATH)
    row = {
        "SNo": next_sno_today,
        "SUC": suc,
        "Name": name,
        "Class": cls,
    }
    next_sno_today += 1
    attendance_buffer.append(row)

# Marked-today set for duplicates (by Name__Roll)
def load_marked_today_keys():
    marked = set()
    path = _day_path()
    if not os.path.exists(path):
        return marked
    try:
        df = pd.read_excel(path)
        if not df.empty:
            for _, r in df.iterrows():
                key = f"{str(r.get('Name',''))}__"  # roll may be unknown; dedupe by name only to be safe
                marked.add(key)
    except Exception:
        pass
    return marked


def save_face_image(image_bgr, face_location, destination_dir, filename_base):
    top, right, bottom, left = face_location
    h, w = image_bgr.shape[:2]
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)
    face_crop = image_bgr[top:bottom, left:right]

    i = 1
    save_name = f"{filename_base}.jpg"
    save_path = os.path.join(destination_dir, save_name)
    while os.path.exists(save_path):
        save_name = f"{filename_base}_{i}.jpg"
        save_path = os.path.join(destination_dir, save_name)
        i += 1

    cv2.imwrite(save_path, face_crop)
    return save_path


def prompt_and_register_unknown(frame_bgr, face_location, known_encodings, known_names):
    print("Register student (press Enter to skip any field):")
    try:
        name = input("Name: ").strip()
        roll_number = input("Roll Number: ").strip()
        class_name = input("Class: ").strip()
        suc_number = input("SUC Number: ").strip()
    except EOFError:
        print("Skipping registration: no interactive input available.")
        return False

    if not name:
        print("Registration cancelled: Name is required.")
        return False

    label = f"{name}__{roll_number}" if roll_number else name

    saved_path = save_face_image(frame_bgr, face_location, KNOWN_FACES_DIR, label)

    try:
        image = face_recognition.load_image_file(saved_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(label)
            try:
                signature = _known_faces_state_signature(KNOWN_FACES_DIR)
                with open(ENCODINGS_CACHE, "wb") as f:
                    pickle.dump({
                        "signature": signature,
                        "encodings": known_encodings,
                        "names": known_names,
                    }, f)
            except Exception:
                pass
        else:
            print("Warning: could not extract face encoding from saved image.")
    except Exception as e:
        print(f"Warning: failed to process saved face image: {e}")

    save_registration_to_excel(name, roll_number, class_name, suc_number, EXCEL_PATH)
    print(f"Registered {label} and saved to '{EXCEL_PATH}'.")
    return True


known_face_encodings, known_face_names = load_known_faces_from_cache_or_build(KNOWN_FACES_DIR, ENCODINGS_CACHE)
print(f"Loaded {len(known_face_encodings)} known face(s) from '{KNOWN_FACES_DIR}'.")

init_attendance_state()
marked_today = load_marked_today_keys()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
try:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video.set(cv2.CAP_PROP_FOURCC, fourcc)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

last_prompt_time = 0.0
frame_count = 0

last_unknown_face_location = None
last_unknown_snapshot = None

# After previous definitions (keep as-is), insert persistence store
last_drawn_faces = []  # list of dicts: {box:(t,r,b,l), label:str, age:int}
MAX_AGE = 10

# Replace earlier occurrences of last_drawn_faces initialization with the above
# and adjust draws to age/persist.

# In the main loop, after display_frame = frame.copy(), keep existing logic but:
# 1) When drawing new detections, store with age=0
# 2) On frames without tracker/detection, redraw aged boxes and increment age
# 3) Drop boxes older than MAX_AGE

# Modify recognition decision function inline where we decide name_label

# Default to faster tracker first
tracker = None
for make in [
    lambda: cv2.TrackerKCF_create(),
    lambda: cv2.TrackerMOSSE_create(),
    lambda: cv2.TrackerCSRT_create(),
]:
    try:
        tracker = make()
        break
    except Exception:
        continue
tracking_active = False
tracked_label = None
last_face_seen_time = 0.0
last_bbox_area = None
# Track the frame shape used at tracker init to avoid OpenCV update mismatches
tracker_init_shape = None

confirm_counts = {}

# Cache tracker bbox per frame
current_bbox = None
tracker_ok_this_frame = False

# FPS and adaptive control
ema_fps = TARGET_FPS
last_loop_time = time.time()
last_adapt_time = time.time()
current_downscale = DOWNSCALE
process_every_n_frames = PROCESS_EVERY_N_FRAMES

# HUD helper
def draw_hud(img, lines, x=10, y=10):
    pad = 6
    line_h = 20
    width = 0
    for t in lines:
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        width = max(width, tw)
    height = line_h * len(lines) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width + pad * 2, y + height), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    yy = y + pad + 15
    for t in lines:
        cv2.putText(img, t, (x + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        yy += line_h

# Add last_detection_time
last_detection_time = 0.0

# Simple KNN-style classification over known encodings
def classify_face_knn(face_encoding, known_encodings, known_names):
    if not known_encodings:
        return "Unknown", 0.0
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    if len(distances) == 0:
        return "Unknown", 0.0
    # k nearest
    k = min(KNN_K, len(distances))
    nn_idx = np.argpartition(distances, k - 1)[:k]
    # Aggregate by label (average distance)
    label_to_dists = {}
    for idx in nn_idx:
        label = known_names[idx]
        d = float(distances[idx])
        label_to_dists.setdefault(label, []).append(d)
    label_avgs = {lab: float(np.mean(ds)) for lab, ds in label_to_dists.items()}
    # Choose best and second-best avg
    ordered = sorted(label_avgs.items(), key=lambda x: x[1])
    best_label, best_dist = ordered[0]
    second_dist = ordered[1][1] if len(ordered) > 1 else 1.0
    # Accept only if within tolerance and with margin
    if (best_dist < RECOGNITION_TOLERANCE) and ((second_dist - best_dist) >= RECOGNITION_MARGIN):
        # Confidence heuristic: map distance to [0..1]
        conf = max(0.0, min(1.0, 1.0 - (best_dist / 0.6)))
        return best_label, conf
    return "Unknown", 0.0

# Smoothing: push and compute majority label and avg confidence
def smooth_label(new_label, new_conf):
    label_history.append((new_label, new_conf))
    # Count labels excluding Unknown
    counts = {}
    conf_sum = {}
    for lab, c in label_history:
        if lab == "Unknown":
            continue
        counts[lab] = counts.get(lab, 0) + 1
        conf_sum[lab] = conf_sum.get(lab, 0.0) + c
    if not counts:
        return "Unknown", 0.0
    # Majority by count, tie-break by avg confidence
    best_lab = max(counts.keys(), key=lambda l: (counts[l], conf_sum[l] / max(1, counts[l])))
    avg_conf = conf_sum[best_lab] / max(1, counts[best_lab])
    # Require majority threshold
    if counts[best_lab] >= (SMOOTH_WINDOW // 2):
        return best_lab, avg_conf
    return "Unknown", 0.0

# Fallback: best-only distance accept
def fallback_best_only(face_encoding, known_encodings, known_names):
    if not known_encodings:
        return "Unknown", 0.0
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    if len(distances) == 0:
        return "Unknown", 0.0
    idx = int(np.argmin(distances))
    d = float(distances[idx])
    if d < FALLBACK_TOLERANCE:
        conf = max(0.0, min(1.0, 1.0 - (d / 0.6)))
        return known_names[idx], conf
    return "Unknown", 0.0

# High-quality recheck on full-res ROI
def recheck_fullres_roi(frame_bgr, box, known_encodings, known_names):
    top, right, bottom, left = box
    h, w = frame_bgr.shape[:2]
    top = max(0, top); left = max(0, left)
    bottom = min(h, bottom); right = min(w, right)
    roi = frame_bgr[top:bottom, left:right]
    if roi.size == 0:
        return "Unknown", 0.0
    # Apply CLAHE on L channel in LAB space to normalize lighting
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    roi_eq = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(roi_eq, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if not encs:
        return "Unknown", 0.0
    enc = encs[0]
    lab, conf = classify_face_knn(enc, known_encodings, known_names)
    if lab == "Unknown":
        lab, conf = fallback_best_only(enc, known_encodings, known_names)
    return lab, conf

# Utility: clamp box to frame bounds
def _clamp_box_tlrb(top, left, right, bottom, frame_shape):
    h, w = frame_shape[:2]
    top = max(0, min(int(top), max(0, h - 1)))
    left = max(0, min(int(left), max(0, w - 1)))
    right = max(left + 1, min(int(right), w))
    bottom = max(top + 1, min(int(bottom), h))
    return top, left, right, bottom


while True:
    ret, frame = video.read()
    if not ret:
        break

    now = time.time()
    dt = max(1e-6, now - last_loop_time)
    inst_fps = 1.0 / dt
    ema_fps = (1 - FPS_SMOOTHING) * ema_fps + FPS_SMOOTHING * inst_fps
    last_loop_time = now

    if (now - last_adapt_time) > ADAPT_INTERVAL_SEC:
        if ema_fps < TARGET_FPS - 2.0:
            process_every_n_frames = min(MAX_SKIP, process_every_n_frames + 1)
            current_downscale = min(MAX_DOWNSCALE, current_downscale + 0.02)
        elif ema_fps > TARGET_FPS + 2.0:
            process_every_n_frames = max(MIN_SKIP, process_every_n_frames - 1)
            current_downscale = max(MIN_DOWNSCALE, current_downscale - 0.02)
        last_adapt_time = now

    frame_count += 1
    # Only allow heavy detection if either tracking is off OR enough time elapsed
    allow_detection_now = (not tracking_active) and ((now - last_detection_time) > 0.0) or ((now - last_detection_time) > DETECTION_INTERVAL_SEC)
    process_this_frame = allow_detection_now and (frame_count % process_every_n_frames == 0)

    display_frame = frame.copy()

    if (len(attendance_buffer) >= BUFFER_MIN_ROWS) or ((time.time() - last_flush_time) > BUFFER_MAX_WAIT_SEC):
        flush_attendance_buffer()

    if tracking_active and (time.time() - last_face_seen_time) > TRACK_TIMEOUT_SEC:
        tracking_active = False
        tracked_label = None
        last_drawn_faces = []

    current_bbox = None
    tracker_ok_this_frame = False
    if tracking_active and tracker is not None:
        if tracker_init_shape is not None and display_frame.shape != tracker_init_shape:
            tracking_active = False
            tracked_label = None
        else:
            try:
                ok, bbox = tracker.update(display_frame)
            except Exception:
                ok = False
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                if w > 0 and h > 0:
                    area = w * h
                    if last_bbox_area is not None:
                        ratio = max(area, last_bbox_area) / max(1, min(area, last_bbox_area))
                        if ratio > AREA_CHANGE_MAX_RATIO:
                            tracking_active = False
                        else:
                            current_bbox = (x, y, w, h)
                            tracker_ok_this_frame = True
                            last_bbox_area = area
                            last_face_seen_time = time.time()
                else:
                    tracking_active = False
            else:
                tracking_active = False

    if tracker_ok_this_frame and current_bbox is not None:
        x, y, w, h = current_bbox
        top, left, right, bottom = y, x, x + w, y + h
        top, left, right, bottom = _clamp_box_tlrb(top, left, right, bottom, display_frame.shape)
        color = (0, 255, 0) if tracked_label != "Unknown (press F)" else (0, 0, 255)
        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(tracked_label or "", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Prefer drawing label above the box; if not enough space, draw below inside bounds
        label_top = max(0, top - text_h - 10)
        label_bottom = label_top + text_h + 10
        if label_top < 0:
            label_top = min(display_frame.shape[0] - (text_h + 10), bottom)
            label_bottom = label_top + text_h + 10
        cv2.rectangle(display_frame, (left, label_top), (min(display_frame.shape[1] - 1, left + text_w + 6), label_bottom), color, -1)
        cv2.putText(display_frame, tracked_label or "", (left + 3, min(display_frame.shape[0] - 1, label_top + text_h + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if process_this_frame and not tracker_ok_this_frame:
        h, w = frame.shape[:2]
        if tracker_ok_this_frame and current_bbox is not None:
            x, y, bw, bh = current_bbox
            margin_x = int(bw * ROI_MARGIN)
            margin_y = int(bh * ROI_MARGIN)
            roi_left = max(0, x - margin_x)
            roi_top = max(0, y - margin_y)
            roi_right = min(w, x + bw + margin_x)
            roi_bottom = min(h, y + bh + margin_y)
        else:
            roi_left, roi_top, roi_right, roi_bottom = 0, 0, w, h

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        small_frame = cv2.resize(roi, (0, 0), fx=current_downscale, fy=current_downscale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Force no upsample for speed; fallback to slight upsample only if nothing found
        face_locations_small = face_recognition.face_locations(
            rgb_small_frame,
            number_of_times_to_upsample=0,
            model=FACE_DETECTION_MODEL,
        )
        if not face_locations_small:
            face_locations_small = face_recognition.face_locations(
                rgb_small_frame,
                number_of_times_to_upsample=1,
                model=FACE_DETECTION_MODEL,
            )
        face_encodings_small = face_recognition.face_encodings(rgb_small_frame, face_locations_small)

        faces_scaled = []
        for (top, right, bottom, left) in face_locations_small:
            top = int(top / current_downscale) + roi_top
            right = int(right / current_downscale) + roi_left
            bottom = int(bottom / current_downscale) + roi_top
            left = int(left / current_downscale) + roi_left
            faces_scaled.append((top, right, bottom, left))

        def center_distance(face):
            t, r, b, l = face
            cx = (l + r) // 2
            cy = (t + b) // 2
            return (cx - display_frame.shape[1] // 2) ** 2 + (cy - display_frame.shape[0] // 2) ** 2

        selected = []
        if faces_scaled:
            order = sorted(range(len(faces_scaled)), key=lambda i: center_distance(faces_scaled[i]))
            idx = order[0]
            selected.append((faces_scaled[idx], face_encodings_small[idx]))

        last_unknown_face_location = None
        last_unknown_snapshot = None
        last_drawn_faces = []

        if not selected:
            tracked_label = None
            tracking_active = False
        else:
            last_detection_time = now

        for (top, right, bottom, left), face_encoding in selected:
            name_label, conf = classify_face_knn(face_encoding, known_face_encodings, known_face_names)
            name_label, conf = smooth_label(name_label, conf)
            if name_label == "Unknown":
                # Fallback on the small-frame encoding
                name_label, conf = fallback_best_only(face_encoding, known_face_encodings, known_face_names)
                if name_label == "Unknown":
                    # High-quality recheck at full-res ROI
                    name_label, conf = recheck_fullres_roi(frame, (top, right, bottom, left), known_face_encodings, known_face_names)

            color = (0, 255, 0) if name_label != "Unknown" else (0, 0, 255)
            # Clamp box
            top, left, right, bottom = _clamp_box_tlrb(top, left, right, bottom, display_frame.shape)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            label_text = name_label if name_label != "Unknown" else ("Unknown (press F)" if MODE == 0 else "Unknown")
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Try to draw below box; if it goes off-screen, draw above
            draw_below = bottom + text_h + 10 <= display_frame.shape[0]
            if draw_below:
                rect_top = bottom
                rect_bottom = bottom + text_h + 10
            else:
                rect_bottom = max(top, text_h + 10)
                rect_top = rect_bottom - (text_h + 10)
            rect_right = min(display_frame.shape[1] - 1, left + text_w + 6)
            cv2.rectangle(display_frame, (left, rect_top), (rect_right, rect_bottom), color, -1)
            text_y = rect_top + text_h + 3 if draw_below else rect_top + text_h - 2
            text_y = min(display_frame.shape[0] - 1, max(0, text_y))
            cv2.putText(display_frame, label_text, (left + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            last_drawn_faces.append((top, right, bottom, left, label_text))

            bbox = (left, top, right - left, bottom - top)
            try:
                tracker = None
                for make in [
                    lambda: cv2.TrackerKCF_create(),
                    lambda: cv2.TrackerMOSSE_create(),
                    lambda: cv2.TrackerCSRT_create(),
                ]:
                    try:
                        tracker = make()
                        break
                    except Exception:
                        continue
                if tracker is not None:
                    tracker.init(display_frame, bbox)
                    tracking_active = True
                    tracked_label = label_text
                    last_face_seen_time = time.time()
                    last_bbox_area = (right - left) * (bottom - top)
                    current_bbox = (left, top, right - left, bottom - top)
                    tracker_ok_this_frame = True
                    tracker_init_shape = display_frame.shape
            except Exception:
                tracking_active = False
                tracked_label = None
                tracker_init_shape = None

            if name_label == "Unknown":
                last_unknown_face_location = (top, right, bottom, left)
                last_unknown_snapshot = frame.copy()
            else:
                key = name_label if "__" in name_label else f"{name_label}__"
                prev = confirm_counts.get(key, 0)
                confirm_counts[key] = prev + 1
                if MODE == 1 and confirm_counts[key] >= MULTIFRAME_CONFIRM and key not in marked_today:
                    save_attendance_buffered(name_label)
                    marked_today.add(key)
                    print(f"Attendance marked: {name_label}")
        for k in list(confirm_counts.keys()):
            if not selected:
                confirm_counts[k] = 0

    if tracking_active and tracker_ok_this_frame and (frame_count % REVALIDATE_EVERY_N_FRAMES == 0):
        try:
            x, y, w, h = current_bbox
            top, right, bottom, left = y, x + w, y + h, x
            rgb_full = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_full, [(top, right, bottom, left)])
            if encs:
                enc = encs[0]
                new_label, conf = classify_face_knn(enc, known_face_encodings, known_face_names)
                new_label, conf = smooth_label(new_label, conf)
                if new_label == "Unknown":
                    tracking_active = False
                    tracked_label = None
                    last_drawn_faces = []
                    tracker_init_shape = None
                else:
                    tracked_label = new_label
                    last_face_seen_time = time.time()
            else:
                tracking_active = False
                tracked_label = None
                last_drawn_faces = []
                tracker_init_shape = None
        except Exception:
            tracking_active = False
            tracked_label = None
            last_drawn_faces = []
            tracker_init_shape = None

    # HUD overlay: add a title line
    mode_text = "Mode: Enroll (0)" if MODE == 0 else "Mode: Attendance (1)"
    controls = "F/E: Register/Add Sample | Q: Quit" if MODE == 0 else "Q: Quit"
    hud_lines = [
        "Smart Attendance - Face Recognition",
        mode_text,
        f"FPS: {ema_fps:.1f} (target {TARGET_FPS:.0f})",
        f"Downscale: {current_downscale:.2f}  Skip: {process_every_n_frames}",
        controls,
    ]
    draw_hud(display_frame, hud_lines, 10, 10)

    cv2.imshow("Face Recognition", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('f') and MODE == 0 and last_unknown_face_location is not None and last_unknown_snapshot is not None:
        prompt_and_register_unknown(last_unknown_snapshot, last_unknown_face_location, known_face_encodings, known_face_names)
        last_unknown_face_location = None
        last_unknown_snapshot = None
    # E: add extra sample for the current tracked label
    if key == ord('e') and MODE == 0 and tracker_ok_this_frame and current_bbox is not None and tracked_label:
        base_label = _parse_label_from_tracked(tracked_label)
        if base_label and base_label != "Unknown (press F)" and base_label != "Unknown":
            if add_sample_for_label(frame, current_bbox, base_label, known_face_encodings, known_face_names):
                print(f"Added extra sample for {base_label}")
    # B: print nearest distances for debug
    if key == ord('b') and tracker_ok_this_frame and current_bbox is not None:
        x, y, w, h = current_bbox
        rgb_full = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb_full, [(y, x + w, y + h, x)])
        if encs and known_face_encodings:
            dists = face_recognition.face_distance(known_face_encodings, encs[0])
            order = np.argsort(dists)[:5]
            print("Nearest distances:")
            for i in order:
                print(f"  {known_face_names[i]}: {dists[i]:.3f}")

flush_attendance_buffer()

video.release()
cv2.destroyAllWindows()
