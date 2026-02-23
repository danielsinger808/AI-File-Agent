# AI-File-Agent
An event-driven Python agent that uses Large Language Models to semantically organize and summarize local files in real-time.

What it does
This agent monitors a local directory for new or modified files and applies "intelligent" logic to determine where they belong. Instead of relying solely on file extensions, it uses an autonomous reasoning loop to read file contents and route them into high-level categories like School, Work, or Finance.

Key Features
Reactive Monitoring: Built on watchdog to handle real-time filesystem events (created, moved, or modified).
LLM-Powered Routing: Uses OpenAI's API to perform content-aware classification for .txt and .md files.
On-Demand Summarization: If a filename contains @sum, the agent automatically generates a concise summary (bullets and tags) as a sidecar file.
Windows-Optimized: Includes specific "wait-until-ready" logic and debouncing to prevent crashes caused by Windows file-access locks during active writes.
Audit Trail: Generates a file_watcher_log.jsonl to track every action, movement, and AI decision for full transparency.

How it works
Trigger: A file is saved or downloaded into the watched folder.
Validation: The agent waits for the file to unlock and stabilize in size.
Analysis: The agent reads a text preview and sends it to the model with a strict routing prompt.
Action: The agent moves the file to its designated category or generates a summary if requested.

Variants
file_watcher_agent.py is the final version, using openai to read and deal with the files. file_watcher_local.py uses a local ai off your machine for these actions. file_watcher.py uses no ai, and just reads text files.

Technical Setup
Python: 3.10+

Dependencies: watchdog, openai, pathlib

Configuration: Simply set your WATCH_DIR and OPENAI_API_KEY to start the loop.
