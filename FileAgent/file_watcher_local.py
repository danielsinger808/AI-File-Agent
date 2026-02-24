import time                 
import json                 
from pathlib import Path    
from watchdog.observers import Observer                     
from watchdog.events import FileSystemEventHandler          
import shutil            
import threading            
from transformers import pipeline                           


# Folder you watch for new/changed files (root "inbox" folder you drop things into)
WATCH_DIR = Path(r"C:\Users\danie\Documents\filetest")

# JSONL log file (one JSON object per line) stored next to this script
LOG_FILE = Path(__file__).with_name("file_watcher_log.jsonl")

WATCH_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".pdf"}  # None to watch all

# This model can "choose" among candidate labels without you training it.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Folder -> list of short descriptions (candidates) that represent that folder.
TXT_LABELS = {
    "School": ["homework", "class notes", "exam study", "lab writeup", "lecture notes"],
    "Work": ["work task", "internship notes", "meeting notes", "project update", "client notes"],
    "Personal": ["personal journal", "plans with friends", "random thoughts", "life note"],
    "Finance": ["bill", "receipt", "budget note", "payment reminder"],
    "Other": ["miscellaneous note"]
}



TXT_AI_FOLDERS = {
    "school": "School",
    "work": "Work",
    "personal": "Personal",
    "receipt": "Receipts",
    "code": "Code",
    "misc": "Misc",
}


# Simple extension-based routing for non-.txt files 
ORGANIZE_MAP = {
    ".pdf": "PDFs",
    ".txt": "Docs",
    ".md": "Docs",
    ".csv": "Data",
    ".log": "Logs",
}

# Debounce prevents the same file from being processed repeatedly in quick succession
DEBOUNCE_SECONDS = 0.5
last_handled = {}   # path -> last timestamp handled


def should_watch(path: Path) -> bool:
    # Ignore folders; only process files
    if path.is_dir():
        return False

    # If None, accept any file
    if WATCH_EXTENSIONS is None:
        return True

    # Otherwise only accept listed extensions
    return path.suffix.lower() in WATCH_EXTENSIONS

#manual unused classification
def classify_txt_light(text: str) -> str:
   
    t = text.lower()

    # School-ish keywords
    if any(k in t for k in ["canvas", "assignment", "midterm", "lecture", "homework", "osu", "cse", "ece"]):
        return "school"

    # Work-ish keywords
    if any(k in t for k in ["meeting", "jira", "ticket", "manager", "deadline", "sprint"]):
        return "work"

    # Receipt/finance-ish keywords
    if any(k in t for k in ["$","total","subtotal","tax","invoice","order","payment","receipt"]):
        return "receipt"

    # Code-ish keywords (basic heuristics)
    if any(k in t for k in ["def ", "class ", "import ", "{", "}", "function", "SELECT ", "FROM "]):
        return "code"

    # Fallback category
    return "misc"


def read_text_preview(path: Path, max_chars: int = 3000) -> str:
    # Read a chunk safely (prevents huge files from being fully loaded)
    # errors="ignore" avoids crashing 
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        # If file can't be read (locked, deleted, encoding issue), return empty string
        return ""


def ai_route_txt(path: Path) -> str:
    
   # Returns folder name like 'School', 'Work', etc.
    #Uses an actual model to decide based on meaning.
    
    # Pull a short preview of the file so inference is fast
    text = read_text_preview(path)

    # If empty/whitespace-only, treat as uncategorizable
    if not text.strip():
        return "Other"

    # Flatten candidate labels into one list of short descriptions (what the model ranks)
    candidates = []
    label_to_folder = {}  # maps each description -> its folder
    for folder, descs in TXT_LABELS.items():
        for d in descs:
            candidates.append(d)
            label_to_folder[d] = folder

    # Run zero-shot: returns ranked labels + scores
    result = classifier(text, candidates)

    # Top predicted "description"
    best_desc = result["labels"][0]

    # Convert that description back into your folder name
    folder = label_to_folder[best_desc]

    # Optional confidence threshold: if model isn't confident, dump into Other
    score = float(result["scores"][0])
    if score < 0.45:
        return "Other"

    return folder


def retry_later(path: Path, seconds: float = 1.0):
    
  #  If a file is locked/busy (common on Windows right after creation/download),
   # this schedules a background retry after a delay.
    
    def _job():
        time.sleep(seconds)
        if path.exists():  # still there (wasn't deleted)
            on_trigger("retry", path)

    # Daemon thread so it won't prevent the program from exiting
    threading.Thread(target=_job, daemon=True).start()


def log_event(event_type: str, path: Path):
    
   # Append an event record to the JSONL log file and print it.
   # JSONL is great because you can stream/grep/parse it later.
    
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),  # human-readable timestamp
        "event": event_type,                       # created/modified/moved/deleted/retry/etc
        "path": str(path),                         # file path string
    }

    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Append a single JSON object per line
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Also print for realtime visibility
    print(record)


def wait_until_ready(path: Path, attempts: int = 20, delay: float = 0.25) -> bool:
    
   # Tries to avoid WinError 32 by waiting until the file is readable and stable.

  #  Strategy:
   # 1) Try opening file for read (detects locks).
   # 2) Wait until file size stops changing (helps with downloads/copies).
    
    last_size = -1

    for _ in range(attempts):
        # If file disappeared, nothing to do
        if not path.exists():
            return False

        try:
            # 1) Try opening for read to detect locks
            with path.open("rb"):
                pass

            # 2) Check size stability (download/copy completion heuristic)
            size = path.stat().st_size
            if size == last_size:
                return True
            last_size = size

        except PermissionError:
            # Still locked by another process
            pass

        time.sleep(delay)

    # Gave up waiting
    return False


def move_safely(src: Path, dest_dir: Path) -> Path:
    
  #  Move src into dest_dir.
  #  If a file with same name already exists, auto-rename with (1), (2), etc.
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    # Auto-rename if name exists
    if dest.exists():
        stem, suf = src.stem, src.suffix
        i = 1
        while True:
            candidate = dest_dir / f"{stem} ({i}){suf}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    # shutil.move handles cross-volume moves too
    shutil.move(str(src), str(dest))
    return dest


def organize_file(path: Path):
    
  #  Main routing logic.
  #  - Only organizes files placed directly in WATCH_DIR (root).
  #  - .txt gets AI-based routing (ai_route_txt)
  #  - Other extensions use ORGANIZE_MAP
    
    ext = path.suffix.lower()

    # only act on files in the root of WATCH_DIR
    # (prevents re-processing files after they’ve been moved into subfolders)
    if path.parent != WATCH_DIR:
        return

    # AI routing for TXT
    if ext == ".txt":
        # Wait for Windows to release the lock / finish writing
        if not wait_until_ready(path):
            print(f"⚠️ TXT still busy/locked: {path.name} (will retry)")
            log_event("busy_retry", path)
            retry_later(path, 1.0)
            return

        # Ask the model which folder to route to
        folder_name = ai_route_txt(path)      # real model decision
        dest_dir = WATCH_DIR / folder_name

        # Move with collision-safe naming
        new_path = move_safely(path, dest_dir)

        print(f"🧠 AI Routed TXT → {folder_name}/")
        log_event("ai_routed_txt", new_path)
        return

    # extension routing for everything else
    folder_name = ORGANIZE_MAP.get(ext)
    if not folder_name:
        # Unknown extension (or not mapped) -> do nothing
        return

    # Wait until the file can be safely moved
    if not wait_until_ready(path):
        print(f"⚠️ File still busy/locked: {path.name} (will retry)")
        log_event("busy_retry", path)
        retry_later(path, 1.0)
        return

    # Move into mapped folder
    dest_dir = WATCH_DIR / folder_name
    new_path = move_safely(path, dest_dir)
    print(f"📂 Moved {path.name} → {folder_name}/")
    log_event("organized", new_path)


def on_trigger(event_type: str, path: Path):
    
   # Central event gate:
   # - Debounces rapid repeat events for the same path
   # - Logs the event
   # - Runs organize_file only for created/moved/retry events
    
    now = time.time()
    last_time = last_handled.get(path)

    # Debounce: ignore duplicate events within the window
    if last_time and (now - last_time) < DEBOUNCE_SECONDS:
        return

    last_handled[path] = now
    log_event(event_type, path)

    try:
        # Only organize on these event types (modified/deleted get logged but not moved)
        if event_type in {"created", "moved", "retry"}:
            organize_file(path)
    except Exception as e:
        # Never kill the watcher thread — swallow exceptions and log them
        print(f"❌ Action failed for {path}: {e}")
        log_event("action_error", path)


class Handler(FileSystemEventHandler):
    
   # Watchdog event handler.
   # Each method is called by watchdog when the OS reports a filesystem event.
    

    def on_created(self, event):
        # New file appeared at event.src_path
        p = Path(event.src_path)
        if should_watch(p):
            on_trigger("created", p)

    def on_modified(self, event):
        # File content changed (often fires multiple times as something writes)
        p = Path(event.src_path)
        if should_watch(p):
            on_trigger("modified", p)

    def on_moved(self, event):
        # File was renamed or moved (destination path is event.dest_path)
        p = Path(event.dest_path)
        if should_watch(p):
            on_trigger("moved", p)

    def on_deleted(self, event):
        # File was deleted (only src_path exists in the event)
        p = Path(event.src_path)
        if should_watch(p):
            on_trigger("deleted", p)


def main():
    # Fail fast if the target directory doesn't exist
    if not WATCH_DIR.exists():
        raise FileNotFoundError(f"WATCH_DIR does not exist: {WATCH_DIR}")

    # Observer is the engine that monitors changes and calls Handler()
    observer = Observer()

    # recursive=True watches subfolders too,
    # but organize_file only processes files in WATCH_DIR root (so moved files won't get reprocessed)
    observer.schedule(Handler(), str(WATCH_DIR), recursive=True)
    observer.start()

    print(f"Watching: {WATCH_DIR} (recursive=True)")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive so watchdog can keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop cleanly on Ctrl+C
        observer.stop()

    # Wait for observer thread to finish
    observer.join()


if __name__ == "__main__":
    # Standard Python entrypoint guard
    main()
