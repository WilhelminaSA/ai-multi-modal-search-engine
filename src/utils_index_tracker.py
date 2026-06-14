import os
import json

TRACK_FILE = r"C:\my_work\AI_MultiModal_Search_Engine\database\file_index.json"


def load_previous_state():
    if not os.path.exists(TRACK_FILE):
        return {}

    with open(TRACK_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(TRACK_FILE, "w") as f:
        json.dump(state, f, indent=4)


def scan_dataset(root_dir):
    image_files = {}

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                full_path = os.path.join(root, f)
                image_files[full_path] = os.path.getmtime(full_path)

    return image_files