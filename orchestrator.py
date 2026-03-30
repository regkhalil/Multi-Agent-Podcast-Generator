import json
import logging
import tomllib
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent / "config.toml"
with open(_config_path, "rb") as f:
    config = tomllib.load(f)


# ── Pydantic output models ──────────────────────────────────────────────────

class PodcastLine(BaseModel):
    speaker: str  # "Ali" (Host) or "Amir" (Guest)
    text: str
    emotion: str


class PodcastScript(BaseModel):
    dialogue: list[PodcastLine]


# ── LLM ─────────────────────────────────────────────────────────────────────

llm = LLM(
    model=f"ollama/{config['ollama']['model']}",
    base_url=config["ollama"]["base_url"],
    temperature=config["ollama"]["temperature"],
)

logger.info("LLM initialized: model=%s, base_url=%s, temperature=%s",
            config["ollama"]["model"], config["ollama"]["base_url"], config["ollama"]["temperature"])


# ── Agents ───────────────────────────────────────────────────────────────────

prompt_expander = Agent(
    role="Prompt Expander",
    goal=(
        "Take a basic topic and expand it into a detailed, multi-faceted "
        "research directive that covers historical, technical, and cultural angles."
    ),
    backstory=(
        "You are a senior research director who excels at turning a simple idea "
        "into a rich, layered brief that guides downstream researchers."
    ),
    llm=llm,
    verbose=True,
)

historian = Agent(
    role="Historian",
    goal="Research the history, origins, and key milestones of the given subject.",
    backstory=(
        "You are a meticulous historian with deep expertise in tracing how ideas "
        "and technologies evolved over time."
    ),
    llm=llm,
    verbose=True,
)

technologist = Agent(
    role="Technologist",
    goal="Research the technical mechanics, engineering specs, and scientific principles of the given subject.",
    backstory=(
        "You are a seasoned technologist who can break down complex engineering "
        "and scientific concepts into clear, accurate explanations."
    ),
    llm=llm,
    verbose=True,
)

futurist = Agent(
    role="Futurist",
    goal="Research future trends, upcoming developments, and pop-culture impact of the given subject.",
    backstory=(
        "You are a forward-thinking futurist who connects emerging trends with "
        "broader societal and cultural implications."
    ),
    llm=llm,
    verbose=True,
)

scriptwriter = Agent(
    role="Scriptwriter",
    goal=(
        "Synthesize research notes into a natural, witty, and engaging podcast "
        "conversation between Ali (the host) and Amir (the guest expert)."
    ),
    backstory=(
        "You are a talented podcast scriptwriter known for sharp dialogue, "
        "smooth transitions, and making complex topics accessible and fun. "
        "Ali is the charismatic host who drives the conversation and asks great "
        "questions. Amir is a knowledgeable guest who provides deep insights "
        "and engaging explanations."
    ),
    llm=llm,
    verbose=True,
)

for agent in [prompt_expander, historian, technologist, futurist, scriptwriter]:
    logger.info("Agent '%s' using model: %s", agent.role, config["ollama"]["model"])


# ── Tasks ────────────────────────────────────────────────────────────────────

expansion_task = Task(
    description=(
        "Take the following basic topic and expand it into a detailed, "
        "2-paragraph master research prompt that covers historical context, "
        "technical details, and cultural/future impact.\n\n"
        "Topic: {topic}"
    ),
    expected_output=(
        "A 2-paragraph research directive that gives clear guidance on the "
        "historical, technical, and cultural/future dimensions to explore."
    ),
    agent=prompt_expander,
)

history_task = Task(
    description=(
        "Using the master research directive provided, research the history "
        "and origins of the subject. Cover key milestones, founding figures, "
        "and pivotal moments."
    ),
    expected_output="Detailed markdown bullet points covering the history and origins.",
    agent=historian,
    context=[expansion_task],
    async_execution=True,
)

tech_task = Task(
    description=(
        "Using the master research directive provided, research the technical "
        "mechanics, engineering specifications, and scientific principles of "
        "the subject."
    ),
    expected_output="Detailed markdown bullet points covering technical mechanics and specs.",
    agent=technologist,
    context=[expansion_task],
    async_execution=True,
)
    
future_task = Task(
    description=(
        "Using the master research directive provided, research future trends, "
        "upcoming developments, and pop-culture impact of the subject."
    ),
    expected_output="Detailed markdown bullet points covering future trends and cultural impact.",
    agent=futurist,
    context=[expansion_task],
    async_execution=True,
)

writing_task = Task(
    description=(
        "Using ALL of the research notes provided, write a lively and engaging "
        "podcast script as a conversation between Ali and Amir. The dialogue "
        "should weave together historical facts, technical details, and future "
        "speculation in a natural, entertaining way. Each line must include a "
        "speaker (either 'Ali' or 'Amir'), the dialogue text, and an emotion.\n\n"
        "TARGET LENGTH: The podcast should be approximately {duration_minutes} minutes long. "
        "Assume ~150 spoken words per minute. Produce enough dialogue lines to fill "
        "the full duration."
    ),
    expected_output=(
        "A complete podcast script in strict JSON format with a 'dialogue' list "
        "where each entry has 'speaker', 'text', and 'emotion' fields."
    ),
    agent=scriptwriter,
    context=[history_task, tech_task, future_task],
    output_pydantic=PodcastScript,
)


# ── Orchestration ────────────────────────────────────────────────────────────

def generate_podcast_script(topic: str, duration_minutes: int = 5) -> str:
    crew = Crew(
        agents=[prompt_expander, historian, technologist, futurist, scriptwriter],
        tasks=[expansion_task, history_task, tech_task, future_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={"topic": topic, "duration_minutes": duration_minutes})

    # result.pydantic holds the validated PodcastScript model
    if result.pydantic:
        script_json = result.pydantic.model_dump_json(indent=2)
    else:
        script_json = str(result)

    # Save to output/script.json for the audio pipeline
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    script_path = output_dir / "script.json"
    script_path.write_text(script_json)
    logger.info("Script saved to %s", script_path)

    return script_json


if __name__ == "__main__":
    output = generate_podcast_script("Agentic AI", duration_minutes=5)
    print(output)
