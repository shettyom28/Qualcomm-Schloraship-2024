# People Counter with YOLOv8 + Firebase

A real-time people counting system that uses a webcam feed, YOLOv8 object detection, and object tracking to count people entering and exiting a defined space. Counts are logged live to a Firebase Firestore database with timestamps.

## How It Works

1. Captures video from a webcam (or a video file).
2. Runs YOLOv8 (`yolov8n.pt`) on each frame to detect people.
3. Tracks each detected person across frames using a custom tracker (`tracker.py`).
4. Two polygon zones (`area1` and `area2`) are defined in the frame. Movement from `area1` → `area2` counts as an entry; the reverse counts as an exit.
5. Entry/exit events, along with a running occupancy count, are written to three Firestore collections: `entering`, `exiting`, and `people`, each with a timestamp (Europe/Dublin timezone).
6. The annotated video feed is displayed live and also saved to `output1.avi`.

## Requirements

- Python 3.8+
- A webcam (or update the script to point to a video file instead)
- A Firebase project with Firestore enabled and a service account key
- Python packages:
  ```bash
  pip install opencv-python pandas numpy ultralytics firebase-admin pytz
  ```
- The YOLOv8 nano weights file: `yolov8n.pt` (downloaded automatically by `ultralytics` on first run, or place it in the project folder)
- A `coco.txt` file listing the COCO class names (one per line), used to filter detections for the `person` class
- A `tracker.py` file defining the `Tracker` class used for multi-object tracking

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/shettyom28/Qualcomm-Schloraship-2024.git
   cd Qualcomm-Schloraship-2024
   ```

2. **Get your Firebase service account key**
   - In the [Firebase Console](https://console.firebase.google.com/), go to your project → **Project Settings** → **Service Accounts** → **Generate New Private Key**.
   - Save the downloaded JSON file somewhere on your machine. **Do not commit this file to the repo** — it grants admin access to your database.

3. **Point the script at your key using an environment variable**
   ```bash
   export FIREBASE_CRED_PATH=/path/to/your-firebase-key.json
   ```
   On Windows (PowerShell):
   ```powershell
   $env:FIREBASE_CRED_PATH="C:\path\to\your-firebase-key.json"
   ```

4. **Check your Firestore database URL**
   - See `database_url.txt` for a link to this project's Firestore console. If you're using your own Firebase project, your URL will differ.

5. **Run the script**
   ```bash
   python finalcode.py
   ```
   Press `Esc` to stop the video feed.

## Project Structure

```
.
├── finalcode.py        # Main detection, tracking, and Firebase logging script
├── tracker.py           # Object tracking logic (not included in this upload — add your own)
├── coco.txt              # COCO class names used for filtering detections
├── database_url.txt    # Link to the Firestore console for this project
└── .gitignore            # Excludes Firebase key files from version control
```

## Security Note

This project requires a Firebase service account key to write to Firestore. That key file should **never** be pushed to GitHub or any public location — it grants full read/write access to the database. Keep it local and reference it only through the `FIREBASE_CRED_PATH` environment variable, as shown above.

## License

Add a license of your choice (e.g. MIT) if you plan to share this project publicly.
