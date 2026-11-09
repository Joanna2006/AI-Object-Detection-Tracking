# AI Object Detection and Tracking System
YOLOv8 + SORT + OpenCV + Streamlit. Detects, labels and tracks objects (unique IDs) from a webcam or uploaded video.

## Run on Windows with VS Code
1. Install Python 3.10–3.12 (tick "Add Python to PATH") and VS Code (with the Python extension).
2. Open the project folder: File > Open Folder > `ai-object-tracking`.
3. Open a terminal (Ctrl+`) and run:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
   (If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)
4. Browser opens at http://localhost:8501. The first run downloads `yolov8n.pt` automatically (internet needed once).
5. Pick Webcam or Upload Video, adjust confidence, press **Start**; **Stop** ends it and enables **Download processed video**.

## Files
- `app.py` dashboard/UI  - `detector.py` YOLOv8 wrapper  - `tracker.py` SORT (Kalman + Hungarian)

## Tips
- Slow? Use `yolov8n.pt`, filter fewer classes, or use a smaller video.
- Webcam fails? Try index 1, and allow camera access in Windows Privacy settings.
