# app.py
import os
import io
import base64
import json
import sqlite3
import calendar
from datetime import datetime, date, timedelta
from functools import wraps
import threading
import tempfile
import pathlib

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import eventlet
eventlet.monkey_patch()

import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import pandas as pd
# FaceNet (facenet-pytorch)
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# ----- CONFIG -----
DB_PATH = 'attendance.db'
SECRET_KEY = 'change_this_secret'  # change to secure secret
RECOGNITION_THRESHOLD = 0.72  # cosine similarity threshold (tuneable)
TOGGLE_WINDOW = 3          # seconds: require this gap before turning a second detection into a clock_out
DEBOUNCE_SECONDS = 3      # seconds: ignore repeated actions within this period
RECOG_STATE = {}
RECOG_LOCK = threading.Lock() 

# ----- INIT APP -----
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# ----- MEDIAPIPE SETUP -----
mp_face_mesh = mp.solutions.face_mesh
face_mesh_processor = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# ----- FACENET / EMBEDDING MODEL -----
# Use InceptionResnetV1 pretrained on VGGFace2 (good general face model)
# Move model to GPU if available, else CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device for FaceNet:", DEVICE)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Preprocessing for facenet: resize to 160x160, to tensor, normalize to [-1,1]
facenet_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# ----- DB HELPERS -----
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            surname TEXT,
            employee_code TEXT UNIQUE,
            created_at TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            embedding TEXT,
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            date TEXT,
            clock_in TEXT,
            clock_out TEXT,
            minutes_in_office INTEGER,
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# formats date and time
def format_time(iso_string):
    """
    Convert ISO datetime string like '2025-10-01T08:13:36' to '08:13'.
    If parsing fails or value is falsy, returns empty string.
    """
    if not iso_string:
        return ''
    try:
        # datetime.fromisoformat handles 'YYYY-MM-DDTHH:MM:SS' format
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%H:%M")
    except Exception:
        # fallback: try a simple string manipulation (in case format varies)
        try:
            s = iso_string.replace('T', ' ')
            return s[11:16]  # best-effort substring
        except Exception:
            return ''

def format_date(iso_string):
    """
    Convert ISO date or datetime string to 'YYYY-MM-DD'.
    If parsing fails or value is falsy, returns empty string.
    """
    if not iso_string:
        return ''
    try:
        # if it's a full datetime, fromisoformat will work; if date-only it also works
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # fallback: if string already looks like 'YYYY-MM-DD...' return first 10 chars
        try:
            return str(iso_string)[:10]
        except Exception:
            return ''

#-------- Last Attendance Helper --------       
def get_last_attendance_state(employee_id):
    """Return 'in' if employee currently clocked in (i.e. last attendance record has NULL clock_out),
       otherwise 'out'."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM attendance WHERE employee_id=? ORDER BY id DESC LIMIT 1', (employee_id,))
    last = cur.fetchone()
    conn.close()
    if last and last['clock_out'] is None:
        return 'in'
    return 'out'

def maybe_toggle_attendance(employee_id):
    """
    Decision function: returns a dict similar to record_attendance()
      - If employee is currently 'out' => do clock_in (unless within debounce)
      - If currently 'in' => if enough time since last_seen (>= TOGGLE_WINDOW) => do clock_out
      - If within DEBOUNCE_SECONDS since last_action => ignore (return action 'none')
    """
    now = datetime.now()
    with RECOG_LOCK:
        st = RECOG_STATE.get(employee_id)
        if not st:
            # initialize from DB
            cur_state = get_last_attendance_state(employee_id)
            st = {'state': cur_state, 'last_action_time': None, 'last_seen': None}
            RECOG_STATE[employee_id] = st

        last_action = st.get('last_action_time')
        last_seen = st.get('last_seen')

        # update last seen timestamp (we saw them right now)
        st['last_seen'] = now

        # Debounce: if last_action was recent, ignore to avoid duplicate writes
        if last_action and (now - last_action).total_seconds() < DEBOUNCE_SECONDS:
            return {'action': 'none', 'time': None, 'minutes': None}

        # If they are currently 'out' in DB/state -> create clock_in immediately
        if st['state'] == 'out':
            res = record_attendance(employee_id)  # will create clock_in
            if res['action'] == 'clock_in':
                st['state'] = 'in'
                st['last_action_time'] = now
            return res

        # If they are 'in', we require a deliberate second detection after TOGGLE_WINDOW:
        # If last_seen exists and time since it is less than TOGGLE_WINDOW, wait (do nothing)
        if last_seen and (now - last_seen).total_seconds() < TOGGLE_WINDOW:
            # not enough time passed between detections -> do nothing (keep waiting)
            return {'action': 'none', 'time': None, 'minutes': None}

        # Enough time has passed -> toggle (clock_out)
        res = record_attendance(employee_id)
        if res['action'] == 'clock_out':
            st['state'] = 'out'
            st['last_action_time'] = now
        return res


# ----- AUTH HELPERS -----
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('login_admin'))
        return f(*args, **kwargs)
    return decorated

def build_filters_from_args(args):
    """
    Reads args: employee_id, month (YYYY-MM), week (YYYY-Www)
    Returns (where_clause, params_list, selected_dict)
    """
    where = "WHERE 1=1"
    params = []
    employee_id = args.get('employee_id') or ''
    month = args.get('month') or ''
    week = args.get('week') or ''

    if employee_id:
        where += " AND e.id = ?"
        params.append(employee_id)

    if month:
        # month is 'YYYY-MM'
        try:
            y_str, m_str = month.split('-')
            y = int(y_str); m = int(m_str)
            first = date(y, m, 1)
            last_day = calendar.monthrange(y, m)[1]
            last = date(y, m, last_day)
            where += " AND a.date BETWEEN ? AND ?"
            params.append(first.isoformat()); params.append(last.isoformat())
        except Exception:
            # ignore bad format
            pass

    if week:
        # week is 'YYYY-Www' (HTML week input format)
        try:
            if '-W' in week:
                y_str, w_str = week.split('-W')
            else:
                # fallback
                parts = week.split('W')
                y_str = parts[0].rstrip('-')
                w_str = parts[-1]
            y = int(y_str); w = int(w_str)
            start = date.fromisocalendar(y, w, 1)  # Monday
            end = date.fromisocalendar(y, w, 7)    # Sunday
            where += " AND a.date BETWEEN ? AND ?"
            params.append(start.isoformat()); params.append(end.isoformat())
        except Exception:
            # ignore bad format
            pass

    selected = {'employee_id': employee_id, 'month': month, 'week': week}
    return where, params, selected

# ----- Dasboard helper -----
def rows_to_dicts(rows):
    """Convert sqlite3.Row objects to plain dicts for Jinja consumption."""
    return [dict(r) for r in rows]


# ----- MEDIAPIPE EMBEDDING FUNCTIONS -----
def image_base64_to_cv2(image_base64_str):
    header, encoded = image_base64_str.split(',', 1) if ',' in image_base64_str else (None, image_base64_str)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_face_embedding_from_bgr(image_bgr):
    """
    Returns a 512-d normalized embedding (list) using facenet (InceptionResnetV1).
    Steps:
      1. Use MediaPipe face landmarks to estimate eye centers and alignment.
      2. Align & crop a square face region, resize to 160x160.
      3. Pass through facenet_model to get embedding, L2-normalize, return as list.
    Returns None if no face detected / error.
    """
    try:
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh_processor.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark

        h, w, _ = img_rgb.shape

        # Landmark indices for approximate eye centers (MediaPipe face mesh)
        # left eye outer/inner ~ 33, 133 ; right eye outer/inner ~ 362, 263
        # We'll average a couple of points for stability.
        left_idxs = [33, 133, 159]   # include upper eyelid idx for stability
        right_idxs = [362, 263, 386]

        left_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in left_idxs], dtype=np.float32)
        right_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in right_idxs], dtype=np.float32)
        left_center = left_pts.mean(axis=0)
        right_center = right_pts.mean(axis=0)

        # compute center between eyes and angle
        eyes_center = (left_center + right_center) / 2.0
        dx, dy = right_center - left_center
        angle = np.degrees(np.arctan2(dy, dx))

        # Build rotation matrix to align eyes horizontally
        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)
        aligned = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR)

        # Recompute landmark positions on rotated image: quick approximate by transforming same points
        ones = np.ones((3, 1))
        # prepare original eye points (use centers) to transform
        pts = np.vstack([left_center, right_center, eyes_center]).T  # shape (2,3)
        # append ones for affine transform
        pts_h = np.vstack([left_center, right_center, eyes_center])
        pts_h = np.hstack([pts_h, np.ones((3,1))])
        transformed = (M @ pts_h.T).T  # shape (3,2)
        t_left = transformed[0]
        t_right = transformed[1]
        t_center = transformed[2]

        # Determine square crop around t_center. Choose size proportional to eye distance.
        eye_dist = np.linalg.norm(t_right - t_left)
        # scale factor to include forehead/chin; tune as needed
        box_size = int(max(160, eye_dist * 4.0))
        cx, cy = int(t_center[0]), int(t_center[1])
        x1 = max(cx - box_size // 2, 0)
        y1 = max(cy - box_size // 2, 0)
        x2 = min(x1 + box_size, aligned.shape[1])
        y2 = min(y1 + box_size, aligned.shape[0])

        crop = aligned[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to PIL Image and apply facenet transform
        pil_img = Image.fromarray(crop)
        inp = facenet_transform(pil_img).unsqueeze(0).to(DEVICE)  # 1x3x160x160

        # forward to get embedding
        with torch.no_grad():
            emb = facenet_model(inp)  # shape [1,512]
        emb_np = emb[0].cpu().numpy()
        # L2-normalize
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm
        return emb_np.tolist()
    except Exception as e:
        print("get_face_embedding_from_bgr error:", e)
        return None


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_employee_embeddings():
    """
    Load per-employee averaged normalized embeddings from DB into a dict {employee_id: np.array}
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT id FROM employees')
    employees = [row['id'] for row in cur.fetchall()]
    emp_map = {}
    for eid in employees:
        cur.execute('SELECT embedding FROM face_embeddings WHERE employee_id=?', (eid,))
        rows = cur.fetchall()
        if not rows:
            continue
        embs = []
        for r in rows:
            try:
                e = np.array(json.loads(r['embedding']), dtype=np.float32)
                if e.size == 0:
                    continue
                embs.append(e)
            except Exception:
                continue
        if not embs:
            continue
        embs = np.stack(embs, axis=0)
        avg = embs.mean(axis=0)
        # normalize
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        emp_map[eid] = avg.tolist()
    conn.close()
    return emp_map


# ----- ATTENDANCE LOGIC -----
def record_attendance(employee_id):
    now = datetime.now()
    today_str = now.date().isoformat()
    conn = get_db()
    cur = conn.cursor()
    # find last attendance for this employee (use ORDER BY id DESC to get most recent)
    cur.execute('SELECT * FROM attendance WHERE employee_id=? ORDER BY id DESC LIMIT 1', (employee_id,))
    last = cur.fetchone()
    if last and last['clock_out'] is None:
        # Set clock_out now and compute minutes
        try:
            clock_in = datetime.fromisoformat(last['clock_in'])
        except Exception:
            # fallback if stored differently
            clock_in = datetime.fromisoformat(str(last['clock_in'])[:19])
        clock_out = now
        minutes = int((clock_out - clock_in).total_seconds() // 60)
        cur.execute('UPDATE attendance SET clock_out=?, minutes_in_office=? WHERE id=?',
                    (clock_out.isoformat(), minutes, last['id']))
        conn.commit()
        conn.close()
        return {'action': 'clock_out', 'time': clock_out.isoformat(), 'minutes': minutes}
    else:
        # Create new attendance record with clock_in and null clock_out
        cur.execute('INSERT INTO attendance (employee_id, date, clock_in, clock_out, minutes_in_office) VALUES (?,?,?,?,?)',
                    (employee_id, today_str, now.isoformat(), None, None))
        conn.commit()
        conn.close()
        return {'action': 'clock_in', 'time': now.isoformat(), 'minutes': None}

# ----- SOCKET.IO: real-time frames -----
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    """
    data: { 'image': 'data:image/jpeg;base64,...' }
    respond: emit 'recognition' with payload {recognized: bool, name: str, surname: str, action: 'clock_in'|'clock_out'|'none', time: iso}
    """
    img_b64 = data.get('image')
    if not img_b64:
        emit('recognition', {'recognized': False, 'label': 'no_image'})
        return
    try:
        img = image_base64_to_cv2(img_b64)
    except Exception as e:
        emit('recognition', {'recognized': False, 'label': 'decode_error'})
        return
    embedding = get_face_embedding_from_bgr(img)
    if embedding is None:
        emit('recognition', {'recognized': False, 'label': 'no_face'})
        return

    emp_embs = load_employee_embeddings()
    best_score = -1.0
    best_emp = None
    for eid, emb in emp_embs.items():
        score = cosine_similarity(embedding, emb)
        if score > best_score:
            best_score = score
            best_emp = eid
    if best_emp is None or best_score < RECOGNITION_THRESHOLD:
        emit('recognition', {'recognized': False, 'label': 'unknown', 'score': best_score})
        return

    # get employee details
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT id, name, surname FROM employees WHERE id=?', (best_emp,))
    emp = cur.fetchone()
    conn.close()
    if not emp:
        emit('recognition', {'recognized': False, 'label': 'unknown_employee_record'})
        return

    # Decide whether to clock in / out using the state machine
    att = maybe_toggle_attendance(best_emp)

    out = {
        'recognized': True,
        'employee_id': int(best_emp),
        'name': emp['name'],
        'surname': emp['surname'],
        'score': best_score,
        'action': att['action'],   # 'clock_in', 'clock_out', or 'none'
        'time': att['time'],
        'minutes': att['minutes']
    }
    emit('recognition', out)


# ----- ROUTES: pages / api -----
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register_admin', methods=['GET', 'POST'])
def register_admin():
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')
        if not all([name, username, password]):
            flash('Fill all fields', 'danger')
            return redirect(url_for('register_admin'))
        pw_hash = generate_password_hash(password)
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute('INSERT INTO admins (name, username, password_hash) VALUES (?,?,?)', (name, username, pw_hash))
            conn.commit()
            flash('Admin registered! Please login.', 'success')
            return redirect(url_for('login_admin'))
        except sqlite3.IntegrityError:
            flash('Username already used', 'danger')
            return redirect(url_for('register_admin'))
        finally:
            conn.close()
    return render_template('register_admin.html')

@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db()
        cur = conn.cursor()
        cur.execute('SELECT * FROM admins WHERE username=?', (username,))
        admin = cur.fetchone()
        conn.close()
        if admin and check_password_hash(admin['password_hash'], password):
            session['admin_id'] = admin['id']
            session['admin_name'] = admin['name']
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
            return redirect(url_for('login_admin'))
    return render_template('login_admin.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('../index')
def index_page():
    # main recognition page (accessible to anyone)
    return render_template('index.html')


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    conn = get_db()
    cur = conn.cursor()

    # Build filters safely if you have build_filters_from_args, otherwise default
    try:
        where, params, selected = build_filters_from_args(request.args)
    except Exception:
        where, params, selected = "", [], {'employee_id':'', 'month':'', 'week':''}

    sql = f"SELECT a.*, e.name as emp_name, e.surname as emp_surname FROM attendance a JOIN employees e ON a.employee_id=e.id {where} ORDER BY a.date ASC, a.id ASC"
    print("[DEBUG] admin_dashboard SQL:", sql, "params:", params)

    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        raw_records = cur.fetchall()
    except Exception as e:
        print("[ERROR] SQL failed:", e)
        raw_records = []

    # Convert sqlite3.Row to dicts
    records = [dict(r) for r in raw_records] if raw_records else []

    # Format date/time fields for display (in-place)
    for r in records:
        # r.get('date') may be 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'
        # store formatted results back in keys to simplify template
        r['date'] = format_date(r.get('date'))
        r['clock_in'] = format_time(r.get('clock_in'))
        r['clock_out'] = format_time(r.get('clock_out'))
        # minutes_in_office left as-is (numeric or None)

    # Employees list for dropdown
    cur.execute('SELECT id, name, surname FROM employees ORDER BY name')
    employees_raw = cur.fetchall()
    employees = [dict(e) for e in employees_raw] if employees_raw else []

    conn.close()

    print(f"[DEBUG] fetched {len(records)} attendance records and {len(employees)} employees")

    qs = request.query_string.decode('utf-8') if request.query_string else ''
    return render_template('admin_dashboard.html', records=records, employees=employees, selected=selected, query_string=qs)


@app.route('/add_employee', methods=['GET', 'POST'])
@login_required
def add_employee():
    """
    Add a new employee OR add more images/videos for an existing employee.
    Accepts:
      - image files via form field name="images"
      - video files via form field name="videos"
    Videos are sampled (default: 1 FPS up to max_frames_per_video) and embeddings are generated.
    """
    emp_id = request.args.get('emp_id')
    conn = get_db()
    cur = conn.cursor()

    employee = None
    if emp_id:
        cur.execute('SELECT id, name, surname, employee_code FROM employees WHERE id=?', (emp_id,))
        employee = cur.fetchone()

    if request.method == 'POST':
        # Form fields
        name = request.form.get('name')
        surname = request.form.get('surname')
        employee_code = request.form.get('employee_code') or (f"{(name or '')[:2]}{(surname or '')[:2]}{int(datetime.now().timestamp())%10000}")

        # Files: images and videos
        image_files = request.files.getlist('images') or []
        video_files = request.files.getlist('videos') or []

        # Basic validation: if adding new employee, require name + surname + at least one file
        if not employee:
            if not name or not surname:
                flash('Please provide name and surname', 'danger')
                return redirect(url_for('add_employee'))
            if not (image_files or video_files):
                flash('Please upload at least one photo or video', 'danger')
                return redirect(url_for('add_employee'))

        # If existing employee - we only need files
        if employee:
            emp_db_id = employee['id']
        else:
            # create new employee row
            try:
                cur.execute(
                    'INSERT INTO employees (name, surname, employee_code, created_at) VALUES (?,?,?,?)',
                    (name, surname, employee_code, datetime.now().isoformat())
                )
                emp_db_id = cur.lastrowid
                conn.commit()
            except sqlite3.IntegrityError:
                conn.close()
                flash('Employee code already exists', 'danger')
                return redirect(url_for('add_employee'))

        # Config: how many frames to extract per video. Tune as needed.
        max_frames_per_video = 10      # limit extracted frames per video
        sample_fps = 1                 # sample rate in frames per second

        saved_embeddings = 0
        errors = 0

        # ---- Process image files (existing behavior, just robustified) ----
        for f in image_files:
            # skip empty file inputs
            if not f or not getattr(f, "filename", ""):
                continue
            try:
                # PIL from stream
                image = Image.open(f.stream).convert('RGB')
                img_cv = pil_to_cv2(image)
                emb = get_face_embedding_from_bgr(img_cv)
                if emb is not None:
                    cur.execute('INSERT INTO face_embeddings (employee_id, embedding) VALUES (?,?)',
                                (emp_db_id, json.dumps(emb)))
                    saved_embeddings += 1
            except Exception as e:
                print('error processing uploaded image', e)
                errors += 1

        # ---- Process video files (NEW) ----
        # We'll write each uploaded video to a temporary file and use OpenCV to sample frames.
        for vf in video_files:
            if not vf or not getattr(vf, "filename", ""):
                continue
            # limit number of videos processed if needed
            try:
                # Create a temporary file to save uploaded video
                suffix = pathlib.Path(vf.filename).suffix or '.mp4'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp_path = tmp.name
                    # Write uploaded bytes to temp file
                    data = vf.read()
                    tmp.write(data)
                    tmp.flush()

                # Open with OpenCV
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    print("Could not open video:", vf.filename)
                    errors += 1
                    try:
                        cap.release()
                    except:
                        pass
                    os.unlink(tmp_path)
                    continue

                # Determine frame sampling step based on video FPS
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                if fps <= 0:
                    # fallback default FPS
                    fps = 25.0
                frame_interval = int(max(1, round(fps / sample_fps)))

                frames_processed = 0
                frame_idx = 0
                while frames_processed < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # only sample every frame_interval frames
                    if frame_idx % frame_interval == 0:
                        # frame is BGR already
                        try:
                            emb = get_face_embedding_from_bgr(frame)
                            if emb is not None:
                                cur.execute('INSERT INTO face_embeddings (employee_id, embedding) VALUES (?,?)',
                                            (emp_db_id, json.dumps(emb)))
                                saved_embeddings += 1
                        except Exception as e:
                            print("error processing video frame:", e)
                            errors += 1
                        frames_processed += 1
                    frame_idx += 1

                # cleanup
                cap.release()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            except Exception as e:
                print("error handling uploaded video file", e)
                errors += 1

        conn.commit()
        conn.close()

        # Flash result
        if saved_embeddings == 0 and errors > 0:
            flash('Uploaded files processed but no faces/embeddings were found in them.', 'warning')
        elif saved_embeddings == 0:
            flash('No faces found in uploaded images/videos. Employee added but without embeddings.', 'warning')
        else:
            if employee:
                flash(f'Added {saved_embeddings} new embeddings to {employee["name"]} {employee["surname"]}', 'success')
            else:
                flash(f'Employee added with {saved_embeddings} embeddings', 'success')

        return redirect(url_for('admin_dashboard'))

    # GET
    return render_template('add_employee.html', employee=employee)


@app.route('/remove_employee', methods=['POST'])
@login_required
def remove_employee():
    emp_id = request.form.get('employee_id')
    if not emp_id:
        flash('No employee selected', 'danger')
        return redirect(url_for('admin_dashboard'))
    conn = get_db()
    cur = conn.cursor()
    cur.execute('DELETE FROM face_embeddings WHERE employee_id=?', (emp_id,))
    cur.execute('DELETE FROM attendance WHERE employee_id=?', (emp_id,))
    cur.execute('DELETE FROM employees WHERE id=?', (emp_id,))
    conn.commit()
    conn.close()
    flash('Employee removed', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/export/excel')
@login_required
def export_excel():
    conn = get_db()
    # respect filters if you implemented build_filters_from_args
    try:
        where, params, _ = build_filters_from_args(request.args)
    except Exception:
        where, params = "", []

    query = f"""
        SELECT a.id, e.name, e.surname, a.date, a.clock_in, a.clock_out, a.minutes_in_office
        FROM attendance a
        JOIN employees e ON a.employee_id=e.id
        {where}
        ORDER BY a.date ASC
    """
    # If using pandas.read_sql_query with params, pass them
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    # Format date/time columns for readability
    if not df.empty:
        df['date'] = df['date'].apply(lambda v: format_date(v) if pd.notna(v) else '')
        df['clock_in'] = df['clock_in'].apply(lambda v: format_time(v) if pd.notna(v) else '')
        df['clock_out'] = df['clock_out'].apply(lambda v: format_time(v) if pd.notna(v) else '')

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='attendance')
    output.seek(0)
    return send_file(output,
                     as_attachment=True,
                     download_name='attendance.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Export to PDF (simple)
@app.route('/export/pdf')
@login_required
def export_pdf():
    conn = get_db()
    try:
        # Respect filters if available
        try:
            where, params, selected = build_filters_from_args(request.args)
        except Exception:
            where, params, selected = "", [], {'employee_id':'', 'month':'', 'week':''}

        query = f"""
            SELECT a.id, e.name, e.surname, a.date, a.clock_in, a.clock_out, a.minutes_in_office
            FROM attendance a
            JOIN employees e ON a.employee_id=e.id
            {where}
            ORDER BY a.date ASC
        """
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    # Format fields
    if not df.empty:
        df['date'] = df['date'].apply(lambda v: format_date(v) if pd.notna(v) else '')
        df['clock_in'] = df['clock_in'].apply(lambda v: format_time(v) if pd.notna(v) else '')
        df['clock_out'] = df['clock_out'].apply(lambda v: format_time(v) if pd.notna(v) else '')
    else:
        # create empty df with expected columns so downstream code stays simple
        df = pd.DataFrame(columns=['id','name','surname','date','clock_in','clock_out','minutes_in_office'])

    # Build PDF using ReportLab Platypus
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=40, rightMargin=40,
                            topMargin=60, bottomMargin=40)

    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1  # center
    meta_style = styles['Normal']
    meta_style.fontSize = 9

    elements = []

    # Title and meta
    title = Paragraph("Attendance report", title_style)
    generated_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_text = f"Generated: {generated_str}"
    # Add info about filters if present
    filters_text = ""
    try:
        sel_emp = selected.get('employee_id')
        sel_month = selected.get('month')
        sel_week = selected.get('week')
        parts = []
        if sel_emp:
            # resolve employee name for nicer display
            try:
                conn2 = get_db()
                cur2 = conn2.cursor()
                cur2.execute("SELECT name, surname FROM employees WHERE id=?", (sel_emp,))
                row = cur2.fetchone()
                if row:
                    parts.append(f"Employee: {row['name']} {row['surname']}")
                conn2.close()
            except Exception:
                parts.append(f"Employee ID: {sel_emp}")
        if sel_month:
            parts.append(f"Month: {sel_month}")
        if sel_week:
            parts.append(f"Week: {sel_week}")
        if parts:
            filters_text = " | ".join(parts)
    except Exception:
        filters_text = ""

    elements.append(title)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(meta_text, meta_style))
    if filters_text:
        elements.append(Paragraph(filters_text, meta_style))
    elements.append(Spacer(1, 12))

    # Table header + data
    headers = ['ID', 'Name', 'Surname', 'Date', 'Clock In', 'Clock Out', 'Minutes']
    data = [headers]

    # rows: convert dataframe rows into lists of strings
    for _, row in df.iterrows():
        rid = row.get('id', '')
        name = row.get('name', '') or ''
        surname = row.get('surname', '') or ''
        date_s = row.get('date', '') or ''
        cin = row.get('clock_in', '') or ''
        cout = row.get('clock_out', '') or ''
        minutes = row.get('minutes_in_office', '')
        minutes_str = str(int(minutes)) if (not pd.isna(minutes) and minutes != None) else ''
        data.append([str(rid), name, surname, date_s, cin, cout, minutes_str])

    # Create table with column widths
    col_widths = [0.6*inch, 1.6*inch, 1.6*inch, 1.1*inch, 0.9*inch, 0.9*inch, 0.8*inch]
    table = Table(data, colWidths=col_widths, repeatRows=1)

    # Table style: header background, grid, alignment, padding
    tbl_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E86AB')),  # header background
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (2,-1), 'LEFT'),  # name and surname left-aligned
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1, -1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 6),
    ])

    # Add subtle alternating row background for readability (after header)
    for i in range(1, len(data)):
        if i % 2 == 0:
            tbl_style.add('BACKGROUND', (0,i), (-1,i), colors.HexColor('#F5F7FA'))

    table.setStyle(tbl_style)

    elements.append(table)

    # Footer with record count
    elements.append(Spacer(1, 12))
    count_text = f"Total records: {len(data)-1}"
    elements.append(Paragraph(count_text, meta_style))

    # Build PDF
    doc.build(elements)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='attendance.pdf', mimetype='application/pdf')


# small API endpoint to list employees (json)
@app.route('/api/employees')
@login_required
def api_employees():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT id, name, surname FROM employees ORDER BY name')
    rows = cur.fetchall()
    conn.close()
    return jsonify([{'id': r['id'], 'name': r['name'], 'surname': r['surname']} for r in rows])

# run
if __name__ == '__main__':
    print("Starting app on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
