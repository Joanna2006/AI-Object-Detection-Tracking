"""AI Object Detection and Tracking System - Streamlit dashboard."""
import os, tempfile, time
from collections import defaultdict
import cv2, numpy as np, pandas as pd, streamlit as st
from detector import Detector
from tracker import Sort

st.set_page_config(page_title="AI Object Detection and Tracking", page_icon="🎯", layout="wide")
st.markdown("""<style>
.stApp{background:#0e1117;color:#e6e6e6}
[data-testid=stSidebar]{background:#161b22;border-right:1px solid #2a2f3a}
[data-testid=stMetric]{background:#161b22;border:1px solid #2a2f3a;border-radius:12px;padding:14px}
.stButton>button,.stDownloadButton>button{width:100%;border-radius:10px;font-weight:600}
h1{background:linear-gradient(90deg,#4f8cff,#9b6bff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
</style>""", unsafe_allow_html=True)

ss = st.session_state
for k, v in {"running": False, "out_path": None}.items():
    ss.setdefault(k, v)


@st.cache_resource
def load_detector(w):
    return Detector(w)


def color_for(i):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def start(): ss.running = True
def stop(): ss.running = False


# ---------------- Sidebar ----------------
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "About"], label_visibility="collapsed")
st.sidebar.divider()
source = st.sidebar.radio("Video source", ["Webcam", "Upload Video"])
upload = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"]) if source == "Upload Video" else None
cam_idx = st.sidebar.number_input("Webcam index", 0, 5, 0) if source == "Webcam" else 0
weights = st.sidebar.selectbox("YOLOv8 model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
det = load_detector(weights)
chosen = st.sidebar.multiselect("Filter classes (empty = all)", list(det.names.values()),
                                default=["person", "car", "bus", "bicycle", "dog"])
c1, c2 = st.sidebar.columns(2)
c1.button("▶ Start", on_click=start, disabled=ss.running)
c2.button("■ Stop", on_click=stop, disabled=not ss.running)

if page == "About":
    st.title("About")
    st.write("Real-time detection with **YOLOv8**, tracking with **SORT** (Kalman filter + Hungarian IoU matching), UI with **Streamlit**.")
    st.stop()

# ---------------- Dashboard ----------------
st.title("AI Object Detection and Tracking System")
video_ph = st.empty()
m = st.columns(5)
ph = [x.empty() for x in m]
table_ph = st.empty()
class_ids = [k for k, v in det.names.items() if v in chosen] or None


def show_stats(fps, active, people, unique, avg_conf):
    for p, (label, val) in zip(ph, [("FPS", f"{fps:.1f}"), ("Objects in frame", active),
                                    ("People", people), ("Unique objects entered", unique),
                                    ("Avg confidence", f"{avg_conf:.0%}")]):
        p.metric(label, val)


if ss.running:
    path = None
    if source == "Upload Video":
        if upload is None:
            st.warning("Please upload a video first."); ss.running = False; st.stop()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(upload.read()); tmp.close()
        path = tmp.name
    cap = cv2.VideoCapture(path if path else int(cam_idx))
    if not cap.isOpened():
        st.error("Could not open video source."); ss.running = False; st.stop()
    W, H = int(cap.get(3)), int(cap.get(4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    ss.out_path = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.mp4")
    writer = cv2.VideoWriter(ss.out_path, cv2.VideoWriter_fourcc(*"mp4v"), min(src_fps, 30), (W, H))
    tracker, seen, fps, n = Sort(), defaultdict(set), 0.0, 0
    try:
        while cap.isOpened() and ss.running:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            tracks = tracker.update(det.detect(frame, conf, class_ids))
            rows = []
            for x1, y1, x2, y2, tid, c, cf in tracks:
                name = det.names[int(c)]
                seen[name].add(int(tid))
                col = color_for(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                label = f"#{int(tid)} {name} {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), col, -1)
                cv2.putText(frame, label, (int(x1) + 3, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                rows.append({"Track ID": int(tid), "Object": name, "Confidence": round(float(cf), 2)})
            fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6) if n else 1 / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            writer.write(frame)
            video_ph.image(frame, channels="BGR", use_container_width=True)
            avg = float(np.mean([r["Confidence"] for r in rows])) if rows else 0.0
            show_stats(fps, len(rows), sum(r["Object"] == "person" for r in rows),
                       sum(len(s) for s in seen.values()), avg)
            if n % 5 == 0:
                table_ph.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            n += 1
    finally:
        cap.release(); writer.release()
    if ss.running:  # video file ended naturally
        ss.running = False
        st.rerun()
    st.session_state["summary"] = {k: len(v) for k, v in seen.items()}
else:
    video_ph.info("Choose a source in the sidebar and press **▶ Start**.")
    show_stats(0, 0, 0, 0, 0)

if not ss.running:
    if ss.get("summary"):
        st.subheader("Detection statistics (unique objects counted)")
        st.bar_chart(pd.Series(ss["summary"], name="Unique objects"))
    if ss.out_path and os.path.exists(ss.out_path):
        with open(ss.out_path, "rb") as f:
            st.download_button("⬇ Download processed video", f, "processed_video.mp4", "video/mp4")
