import time
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import threading
import os
from openai import OpenAI

# --- OpenAI client / routing configuration ---
# Expects OPENAI_API_KEY to be set in your environment (setx OPENAI_API_KEY "...").
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Allowed routing destinations for TXT classification (returned verbatim by the model).
ROUTE_FOLDERS = ["School", "Work", "Personal", "Finance", "Other"]

# --- Watcher configuration ---
WATCH_DIR = Path(r"C:\Users\danie\Documents\filetest")  # folder being monitored
LOG_FILE = Path(__file__).with_name("file_watcher_log.jsonl")  # JSONL audit log next to this script
WATCH_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".pdf"}  # set to None to watch everything

# --- Legacy / leftover mapping structures (not used by the OpenAI routing path below) ---
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

# Extension-based routing for non-TXT files.
ORGANIZE_MAP = {
    ".pdf": "PDFs",
    ".txt": "Docs",
    ".md": "Docs",
    ".csv": "Data",
    ".log": "Logs",
}

# --- Event storm protection ---
DEBOUNCE_SECONDS = 0.5
last_handled = {}  # maps Path -> last timestamp handled (coalesces rapid duplicate events)

def should_watch(path: Path) -> bool:
    """Filter out directories and extensions you don't care about."""
    if path.is_dir():
        return False
    if WATCH_EXTENSIONS is None:
        return True
    return path.suffix.lower() in WATCH_EXTENSIONS

# --- Legacy keyword classifier (not used by the OpenAI routing path below) ---
def classify_txt_light(text: str) -> str:
    """Cheap keyword-based classification; kept for fallback/experiments."""
    t = text.lower()

    if any(k in t for k in ["canvas", "assignment", "midterm", "lecture", "homework", "osu", "cse", "ece"]):
        return "school"
    if any(k in t for k in ["meeting", "jira", "ticket", "manager", "deadline", "sprint"]):
        return "work"
    if any(k in t for k in ["$","total","subtotal","tax","invoice","order","payment","receipt"]):
        return "receipt"
    if any(k in t for k in ["def ", "class ", "import ", "{", "}", "function", "SELECT ", "FROM "]):
        return "code"

    return "misc"

def read_text_preview(path: Path, max_chars: int = 1200) -> str:
    """Read a small slice of a text file (protects against huge files / encoding issues)."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""

def ai_route_txt(path: Path) -> str:
    """
    Use ChatGPT (via OpenAI API) to classify a TXT file into one of ROUTE_FOLDERS.
    Returns a folder name like: School / Work / Personal / Finance / Other.
    """
    text = read_text_preview(path, max_chars=3000).strip()
    if not text:
        return "Other"

    # Minimal, strict prompt: model must return exactly one folder label.
    prompt = f"""
You are a file organizer. Choose exactly ONE folder for this text from:
{ROUTE_FOLDERS}

Rules:
- Return ONLY the folder name (no extra words).
- If unsure, return "Other".

Text:
{text}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0
    )

    folder = resp.output_text.strip()

    # Guardrail: if the model returns something unexpected, default to Other.
    if folder not in ROUTE_FOLDERS:
        return "Other"
    return folder

def retry_later(path: Path, seconds: float = 1.0):
    """If a file is still being written/locked, retry after a short delay in a daemon thread."""
    def _job():
        time.sleep(seconds)
        if path.exists():  # still there
            on_trigger("retry", path)
    threading.Thread(target=_job, daemon=True).start()

def log_event(event_type: str, path: Path, **extra):
    """
    Append a JSON line to LOG_FILE for auditing/debugging.
    Extra keyword args get included (e.g., summary=..., error=...).
    """
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "path": str(path),
        **extra
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(record)

def wait_until_ready(path: Path, attempts: int = 20, delay: float = 0.25) -> bool:
    """
    Windows can hold files open right after creation/download.
    This waits until the file is readable and its size stabilizes to avoid WinError 32.
    """
    last_size = -1
    for _ in range(attempts):
        if not path.exists():
            return False

        try:
            # Try opening for read to detect locks.
            with path.open("rb"):
                pass

            # Wait for size to stop changing (helps for downloads/editor writes).
            size = path.stat().st_size
            if size == last_size:
                return True
            last_size = size

        except PermissionError:
            pass

        time.sleep(delay)

    return False

def move_safely(src: Path, dest_dir: Path) -> Path:
    """
    Move a file into dest_dir.
    If a file with the same name already exists there, auto-rename with (1), (2), ...
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

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

# --- Summarization settings ---
# If False, summaries only run when the filename contains "@sum".
AUTO_SUMMARIZE_TXT = False

def wants_summary(path: Path) -> bool:
    """On-demand trigger: include '@sum' in the filename to request summarization."""
    return "@sum" in path.stem.lower()

def ai_summarize_txt(path: Path) -> str:
    """
    Use ChatGPT (via OpenAI API) to generate a short overview of a TXT file:
    - 3 bullets + 3 tags.
    """
    text = read_text_preview(path, max_chars=4000).strip()
    if not text:
        return ""

    prompt = f"""
Summarize the text below in:
- 3 concise bullet points
- then 3 short tags (single words or short phrases)

Text:
{text}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.2,
    )

    return resp.output_text.strip()

def write_summary_sidecar(final_path: Path, summary: str):
    """Optional helper to save a summary next to the file as '<name>.summary.txt'."""
    summary_path = final_path.with_suffix(final_path.suffix + ".summary.txt")
    summary_path.write_text(summary, encoding="utf-8")
    return summary_path

def organize_file(path: Path):
    """
    Main "act" step:
    - Only processes files that are directly inside WATCH_DIR (not nested folders).
    - TXT: AI route (and optionally summarize).
    - Other extensions: static extension-based routing.
    """
    ext = path.suffix.lower()

    # Only act on files in the root of WATCH_DIR (prevents re-processing moved files).
    if path.parent != WATCH_DIR:
        return

    # --- AI routing for TXT ---
    if ext == ".txt":
        # Avoid classifying/moving files still being written/locked.
        if not wait_until_ready(path):
            print(f"⚠️ TXT still busy/locked: {path.name} (will retry)")
            log_event("busy_retry", path)
            retry_later(path, 1.0)
            return

        folder_name = ai_route_txt(path)
        dest_dir = WATCH_DIR / folder_name
        new_path = move_safely(path, dest_dir)

        print(f"🧠 AI Routed TXT → {folder_name}/")
        log_event("ai_routed_txt", new_path)

        # --- optional summarization ---
        summary = None
        do_summary = AUTO_SUMMARIZE_TXT or wants_summary(path)

        if do_summary:
            try:
                # Summarize after moving so the summary attaches to the final path.
                summary = ai_summarize_txt(new_path)
                log_event("ai_summary_txt", new_path, summary=summary)

                # Optional: write summary to a sidecar file (commented out by default).
                # sp = write_summary_sidecar(new_path, summary)
                # log_event("summary_written", sp)

            except Exception as e:
                log_event("summary_error", new_path, error=str(e))

        return

    # --- extension routing for everything else ---
    folder_name = ORGANIZE_MAP.get(ext)
    if not folder_name:
        return

    if not wait_until_ready(path):
        print(f"⚠️ File still busy/locked: {path.name} (will retry)")
        log_event("busy_retry", path)
        retry_later(path, 1.0)
        return

    dest_dir = WATCH_DIR / folder_name
    new_path = move_safely(path, dest_dir)

    print(f"📂 Moved {path.name} → {folder_name}/")
    log_event("organized", new_path)

def on_trigger(event_type: str, path: Path):
    """
    Debounced entrypoint called by watchdog events.
    Only triggers actions for a subset of event types.
    """
    now = time.time()
    last_time = last_handled.get(path)

    # Coalesce rapid repeats for the same path (common with editor saves/downloads).
    if last_time and (now - last_time) < DEBOUNCE_SECONDS:
        return

    last_handled[path] = now
    log_event(event_type, path)

    try:
        if event_type in {"created", "moved", "retry"}:
            organize_file(path)
    except Exception as e:
        # Protect the observer thread from crashing on unexpected exceptions.
        print(f"❌ Action failed for {path}: {e}")
        log_event("action_error", path)

class Handler(FileSystemEventHandler):
    """Watchdog handler that converts filesystem events into our on_trigger() calls."""
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
    """Bootstraps the watchdog observer and keeps the process alive until Ctrl+C."""
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
