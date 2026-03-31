"""
Streamlit interface for the Multi-Agent Podcast Generator.

Allows users to input a topic, select duration, monitor agent progress,
and download the generated podcast audio.
"""

import json
import logging
import threading
import queue
from pathlib import Path

import streamlit as st

from orchestrator import generate_podcast_script
from audio_pipeline import generate_audio_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Pipeline steps shown in the progress tracker
PIPELINE_STEPS = [
    "Prompt Expansion",
    "Historical Research",
    "Technical Research",
    "Future Trends Research",
    "Script Writing",
    "Audio Synthesis",
]

# Keywords for detecting which step is active from logs
STEP_KEYWORDS = [
    ["prompt expansion", "expand"],
    ["history", "historian", "historical"],
    ["technical", "technologist", "tech_task"],
    ["future", "futurist", "trends"],
    ["script", "scriptwriter", "writing"],
    ["audio", "synthesis", "tts", "stitching"],
]


class LogCapture(logging.Handler):
    """Captures log records and pushes them to a queue for live display."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def run_generation(
    topic: str,
    duration: int,
    log_queue: queue.Queue,
    result_holder: dict,
) -> None:
    """Execute the full pipeline (script + audio) in a background thread."""
    handler = LogCapture(log_queue)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        log_queue.put("Starting script generation...")
        script_json = generate_podcast_script(topic, duration)
        result_holder["script_json"] = script_json

        log_queue.put("Script complete. Starting audio synthesis...")
        audio_path = generate_audio_sync(script_json)
        result_holder["audio_path"] = str(audio_path)

        log_queue.put("Done.")
    except Exception as e:
        result_holder["error"] = str(e)
        log_queue.put(f"ERROR: {e}")
    finally:
        root_logger.removeHandler(handler)
        log_queue.put("__DONE__")


def init_session_state() -> None:
    """Set default values for all session keys."""
    defaults = {
        "generation_started": False,
        "generation_complete": False,
        "script_json": None,
        "audio_path": None,
        "logs": [],
        "error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state() -> None:
    """Clear results so the user can start a new generation."""
    st.session_state.generation_started = False
    st.session_state.generation_complete = False
    st.session_state.script_json = None
    st.session_state.audio_path = None
    st.session_state.logs = []
    st.session_state.error = None


def detect_active_step() -> int:
    """Return the index of the last detected pipeline step from logs."""
    combined = " ".join(st.session_state.logs).lower()
    active = 0
    for i, keywords in enumerate(STEP_KEYWORDS):
        if any(kw in combined for kw in keywords):
            active = i + 1
    return active


def render_pipeline_status() -> None:
    """Show pipeline steps using native Streamlit status indicators."""
    active_index = detect_active_step()
    completed = st.session_state.generation_complete

    for i, step_name in enumerate(PIPELINE_STEPS):
        if completed or i < active_index:
            st.success(step_name, icon=":material/check_circle:")
        elif i == active_index and st.session_state.generation_started:
            st.info(step_name, icon=":material/pending:")
        else:
            st.markdown(f"&nbsp;&nbsp;&nbsp; {step_name}")


def render_script_preview() -> None:
    """Display the script as a chat conversation using st.chat_message."""
    if not st.session_state.script_json:
        return

    try:
        script = json.loads(st.session_state.script_json)
    except json.JSONDecodeError:
        st.code(st.session_state.script_json)
        return

    for line in script.get("dialogue", []):
        speaker = line.get("speaker", "Unknown")
        text = line.get("text", "")

        if speaker == "Ali":
            with st.chat_message("user", avatar=":material/mic:"):
                st.markdown(f"**Ali (Host):** {text}")
        else:
            with st.chat_message("assistant", avatar=":material/headphones:"):
                st.markdown(f"**Amir (Guest):** {text}")


def render_results(topic: str) -> None:
    """Display audio player, download button, and script preview."""
    if st.session_state.error:
        st.error(f"Generation failed: {st.session_state.error}")
        return

    if not st.session_state.generation_complete or not st.session_state.audio_path:
        return

    audio_path = Path(st.session_state.audio_path)
    if not audio_path.exists():
        st.warning("Audio file not found on disk.")
        return

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    st.audio(audio_bytes, format="audio/mp3")

    safe_name = topic.replace(" ", "_")[:30] if topic else "podcast"
    st.download_button(
        label="Download MP3",
        data=audio_bytes,
        file_name=f"podcast_{safe_name}.mp3",
        mime="audio/mp3",
        use_container_width=True,
    )

    with st.expander("View full script", expanded=False):
        render_script_preview()


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Podcast Gen",
        page_icon=":material/podcasts:",
        layout="wide",
    )
    init_session_state()

    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        duration = st.slider("Duration (min)", min_value=2, max_value=15, value=5)

        st.divider()

        generate_btn = st.button(
            "Generate Podcast",
            disabled=st.session_state.generation_started,
            use_container_width=True,
            type="primary",
        )
        reset_btn = st.button(
            "New Podcast",
            disabled=not st.session_state.generation_complete,
            use_container_width=True,
        )

        if reset_btn:
            reset_state()
            st.rerun()

        st.divider()
        # Display the active LLM provider from config
        try:
            import tomllib
            with open(Path(__file__).parent / "config.toml", "rb") as _f:
                _cfg = tomllib.load(_f)
            _provider = _cfg.get("llm", {}).get("provider", "ollama")
            if _provider == "gemini":
                _model = _cfg["gemini"]["model"]
            else:
                _model = _cfg["ollama"]["model"]
            st.caption(f"Provider: {_provider} | Model: {_model}")
        except Exception:
            pass
        st.caption("Agents: Prompt Expander, Historian, Technologist, Futurist, Scriptwriter")

    # Title
    st.title(":material/podcasts: Multi-Agent Podcast Generator")
    st.markdown("Generate a full podcast on any topic using a pipeline of specialized AI agents.")

    # Centered topic input
    spacer_l, center, spacer_r = st.columns([1, 3, 1])
    with center:
        topic = st.text_input(
            "Podcast Topic",
            placeholder="e.g., The History of Quantum Computing",
            label_visibility="collapsed",
        )

    st.divider()

    # Two-column layout: pipeline | output
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.subheader("Pipeline")
        render_pipeline_status()

        with st.expander("Logs (Terminal)", expanded=False):
            if st.session_state.logs:
                st.code("\n".join(st.session_state.logs[-50:]), language="log")
            else:
                st.caption("No activity yet.")

    with col_right:
        st.subheader("Output")

        if st.session_state.generation_complete:
            render_results(topic)
        elif st.session_state.generation_started:
            st.info("Generation in progress...")
        else:
            st.info("Enter a topic above and click **Generate Podcast** in the sidebar.")

    # Trigger generation
    if generate_btn and topic:
        st.session_state.generation_started = True
        st.session_state.logs = []

        log_queue = queue.Queue()
        result_holder = {}

        thread = threading.Thread(
            target=run_generation,
            args=(topic, duration, log_queue, result_holder),
        )
        thread.start()

        # Live progress in a st.status block
        with col_left:
            with st.status("Agents are working...", expanded=True) as status:
                log_area = st.empty()

                while True:
                    try:
                        msg = log_queue.get(timeout=0.5)
                        if msg == "__DONE__":
                            break
                        st.session_state.logs.append(msg)
                        log_area.code(
                            "\n".join(st.session_state.logs[-30:]),
                            language="log",
                        )
                    except queue.Empty:
                        continue

                status.update(label="Generation complete.", state="complete")

        thread.join()

        if "error" in result_holder:
            st.session_state.error = result_holder["error"]
        else:
            st.session_state.script_json = result_holder.get("script_json")
            st.session_state.audio_path = result_holder.get("audio_path")

        st.session_state.generation_started = False
        st.session_state.generation_complete = True
        st.rerun()


if __name__ == "__main__":
    main()
