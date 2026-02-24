import time
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import threading

WATCH_DIR = Path(r"C:\Users\danie\Documents\filetest")
LOG_FILE = Path(__file__).with_name("file_watcher_log.jsonl")

WATCH_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".pdf"}  # None to watch all

ORGANIZE_MAP = {
    ".pdf": "PDFs",
    ".txt": "Docs",
    ".md": "Docs",
    ".csv": "Data",
    ".log": "Logs",
}

DEBOUNCE_SECONDS = 0.5
last_handled = {}


def should_watch(path: Path) -> bool:
    if path.is_dir():
        return False
    if WATCH_EXTENSIONS is None:
        return True
    return path.suffix.lower() in WATCH_EXTENSIONS

def retry_later(path: Path, seconds: float = 1.0):
    def _job():
        time.sleep(seconds)
        if path.exists():  # still there
            on_trigger("retry", path)
    threading.Thread(target=_job, daemon=True).start()


def log_event(event_type: str, path: Path):
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "path": str(path),
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(record)


def wait_until_ready(path: Path, attempts: int = 20, delay: float = 0.25) -> bool:
    """
    Tries to avoid WinError 32 by waiting until the file is readable and stable.
    """
    last_size = -1
    for _ in range(attempts):
        if not path.exists():
            return False

        try:
            # 1) try opening for read to detect locks
            with path.open("rb"):
                pass

            # 2) optional: wait for size to stop changing (helps for downloads)
            size = path.stat().st_size
            if size == last_size:
                return True
            last_size = size

        except PermissionError:
            pass

        time.sleep(delay)

    return False


def move_safely(src: Path, dest_dir: Path) -> Path:
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

    shutil.move(str(src), str(dest))
    return dest


def organize_file(path: Path):
    ext = path.suffix.lower()
    folder_name = ORGANIZE_MAP.get(ext)
    if not folder_name:
        return

    # Don’t re-organize files already inside subfolders
    if path.parent != WATCH_DIR:
        return

    # Wait until file isn't locked / is done writing
    if not wait_until_ready(path):
        print(f"⚠️ Busy/locked, will retry: {path.name}")
        log_event("busy_retry_scheduled", path)
        retry_later(path, seconds=1.0)   # try again in 1s
        return


    dest_dir = WATCH_DIR / folder_name
    new_path = move_safely(path, dest_dir)

    print(f"📂 Moved {path.name} → {folder_name}/")
    log_event("organized", new_path)


def on_trigger(event_type: str, path: Path):
    now = time.time()
    last_time = last_handled.get(path)

    if last_time and (now - last_time) < DEBOUNCE_SECONDS:
        return

    last_handled[path] = now
    log_event(event_type, path)

    try:
        if event_type in {"created", "moved", "retry"}:
            organize_file(path)
    except Exception as e:
        # Never kill the watcher thread
        print(f"❌ Action failed for {path}: {e}")
        log_event("action_error", path)


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        p = Path(event.src_path)
        if should_watch(p):
            on_trigger("created", p)

    def on_modified(self, event):
        p = Path(event.src_path)
        if should_watch(p):
            on_trigger("modified", p)

    def on_moved(self, event):
        p = Path(event.dest_path)
        if should_watch(p):
            on_trigger("moved", p)

    def on_deleted(self, event):
        p = Path(event.src_path)
        if should_watch(p):
            on_trigger("deleted", p)


def main():
    if not WATCH_DIR.exists():
        raise FileNotFoundError(f"WATCH_DIR does not exist: {WATCH_DIR}")

    observer = Observer()
    observer.schedule(Handler(), str(WATCH_DIR), recursive=True)
    observer.start()

    print(f"Watching: {WATCH_DIR} (recursive=True)")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
