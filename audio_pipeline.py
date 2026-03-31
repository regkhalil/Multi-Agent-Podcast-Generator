import asyncio
import json
import logging
import tomllib
from pathlib import Path

import edge_tts
import static_ffmpeg
from pydub import AudioSegment

static_ffmpeg.add_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent / "config.toml"
with open(_config_path, "rb") as f:
    config = tomllib.load(f)

VOICE_MAP = {
    "Ali": config["tts"]["voice_ali"],
    "Amir": config["tts"]["voice_amir"],
}

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


async def _synthesize_line(text: str, voice: str, output_path: Path) -> None:
    """Generate a single TTS audio clip."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


async def generate_audio(script_json: str, output_file: str = "podcast.mp3") -> Path:
    """Convert a PodcastScript JSON string into a stitched MP3 file."""
    script = json.loads(script_json, strict=False)
    lines = script["dialogue"]

    temp_dir = OUTPUT_DIR / "segments"
    temp_dir.mkdir(exist_ok=True)

    # Generate all audio segments
    tasks = []
    segment_paths = []
    for i, line in enumerate(lines):
        speaker = line["speaker"]
        voice = VOICE_MAP.get(speaker)
        if not voice:
            logger.warning("Unknown speaker '%s' at line %d, skipping", speaker, i)
            continue

        seg_path = temp_dir / f"line_{i:04d}.mp3"
        segment_paths.append(seg_path)
        logger.info("Generating line %d/%d — %s: %.50s...", i + 1, len(lines), speaker, line["text"])
        tasks.append(_synthesize_line(line["text"], voice, seg_path))

    await asyncio.gather(*tasks)

    # Stitch segments together with a short pause between speakers
    logger.info("Stitching %d segments...", len(segment_paths))
    pause = AudioSegment.silent(duration=400)  # 400ms pause between lines
    podcast = AudioSegment.empty()

    for seg_path in segment_paths:
        segment = AudioSegment.from_mp3(seg_path)
        podcast += segment + pause

    final_path = OUTPUT_DIR / output_file
    podcast.export(str(final_path), format="mp3")
    logger.info("Podcast saved to %s", final_path)

    # Clean up temp segments
    for seg_path in segment_paths:
        seg_path.unlink(missing_ok=True)
    temp_dir.rmdir()

    return final_path


def generate_audio_sync(script_json: str, output_file: str = "podcast.mp3") -> Path:
    """Synchronous wrapper around generate_audio."""
    return asyncio.run(generate_audio(script_json, output_file))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_pipeline.py <script.json>")
        sys.exit(1)

    script_path = Path(sys.argv[1])
    # Derive audio filename from script filename: script_20260331_120000.json -> podcast_20260331_120000.mp3
    stem = script_path.stem
    audio_name = stem.replace("script_", "podcast_", 1) + ".mp3" if stem.startswith("script_") else "podcast.mp3"
    script_json = script_path.read_text()
    result = generate_audio_sync(script_json, audio_name)
    print(f"Audio saved to: {result}")
