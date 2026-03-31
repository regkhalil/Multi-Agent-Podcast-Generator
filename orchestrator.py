import json
import logging
import os
import tomllib
from datetime import datetime
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv
from pydantic import BaseModel

from research_tools import get_research_tools

# ── Config ───────────────────────────────────────────────────────────────────

load_dotenv()

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

_provider = config.get("llm", {}).get("provider", "ollama")

if _provider == "gemini":
    _gemini_cfg = config["gemini"]
    _api_key = os.environ.get("GEMINI_API_KEY")
    if not _api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required when provider is 'gemini'. "
                           "Set it in a .env file or export it in your shell.")
    llm = LLM(
        model=f"gemini/{_gemini_cfg['model']}",
        api_key=_api_key,
        temperature=_gemini_cfg["temperature"],
    )
    logger.info("LLM initialized: provider=gemini, model=%s, temperature=%s",
                _gemini_cfg["model"], _gemini_cfg["temperature"])
else:
    _ollama_cfg = config["ollama"]
    llm = LLM(
        model=f"ollama/{_ollama_cfg['model']}",
        base_url=_ollama_cfg["base_url"],
        temperature=_ollama_cfg["temperature"],
    )
    logger.info("LLM initialized: provider=ollama, model=%s, base_url=%s, temperature=%s",
                _ollama_cfg["model"], _ollama_cfg["base_url"], _ollama_cfg["temperature"])


# ── Feature flags ────────────────────────────────────────────────────────────

_research_tools = get_research_tools()

logger.info("Research tools loaded: %s", [t.name for t in _research_tools])


# ── Agents ───────────────────────────────────────────────────────────────────

prompt_expander = Agent(
    role="Podcast Producer",
    goal=(
        "Transform a raw topic into a structured, multi-angle research brief "
        "tailored for an engaging podcast episode. Identify what makes the topic "
        "interesting, surprising, and debatable."
    ),
    backstory=(
        "You are a veteran podcast producer who has launched dozens of hit shows. "
        "You know that great episodes start with great briefs — ones that give "
        "researchers specific questions to chase, not vague directions. You always "
        "think about the listener: what hooks them in, what surprises them, and "
        "what keeps them listening. You tailor the research angle to the nature of "
        "the topic — a tech topic gets different treatment than a cultural one."
    ),
    llm=llm,
    verbose=True,
)

_RESEARCH_BACKSTORY_SUFFIX = (
    " If tools are available, search for relevant information first, but always "
    "combine retrieved data with your own expertise. If no relevant results are "
    "found, rely entirely on your own knowledge."
)

historian = Agent(
    role="Historian",
    goal="Research the history, origins, and key milestones of the given subject.",
    backstory=(
        "You are a meticulous historian with deep expertise in tracing how ideas "
        "and technologies evolved over time."
        + _RESEARCH_BACKSTORY_SUFFIX
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
        + _RESEARCH_BACKSTORY_SUFFIX
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
        + _RESEARCH_BACKSTORY_SUFFIX
    ),
    llm=llm,
    verbose=True,
)

critic = Agent(
    role="Research Critic",
    goal=(
        "Evaluate each research contribution on relevance, depth, and accuracy, "
        "then assign a quality score (0-100) and a recommended weighting percentage."
    ),
    backstory=(
        "You are a rigorous editorial director who reviews research drafts before "
        "they go to production. You judge content on factual depth, relevance to the "
        "topic, and overall quality. You are fair but demanding."
    ),
    llm=llm,
    verbose=True,
)

scriptwriter = Agent(
    role="Scriptwriter",
    goal=(
        "Synthesize research notes into a natural, witty, and engaging podcast "
        "conversation between Ali (the host) and Amir (the guest expert). "
        "Allocate dialogue proportionally to the quality ratings from the critic."
    ),
    backstory=(
        "You are a talented podcast scriptwriter known for sharp dialogue, "
        "smooth transitions, and making complex topics accessible and fun. "
        "Ali is the charismatic host who drives the conversation and asks great "
        "questions. Amir is a knowledgeable guest who provides deep insights "
        "and engaging explanations. You pay close attention to the critic's "
        "ratings — higher-rated research gets more airtime in the conversation."
    ),
    llm=llm,
    verbose=True,
)

for agent in [prompt_expander, historian, technologist, futurist, critic, scriptwriter]:
    logger.info("Agent '%s' using provider: %s", agent.role, _provider)


# ── Tasks ────────────────────────────────────────────────────────────────────

expansion_task = Task(
    description=(
        "Take the topic below and produce a structured PODCAST RESEARCH BRIEF.\n\n"
        "Topic: {topic}\n\n"
        "Your brief MUST include these sections:\n\n"
        "1. ONE-LINER: A single sentence capturing what this topic is about.\n\n"
        "2. AUDIENCE HOOK: Why should a general listener care? What's the "
        "surprising, counterintuitive, or high-stakes angle?\n\n"
        "3. HISTORIAN QUESTIONS: 3-4 specific research questions for the "
        "history expert (e.g., 'Who first proposed X and why was it "
        "controversial?', not just 'research the history').\n\n"
        "4. TECHNOLOGIST QUESTIONS: 3-4 specific research questions for the "
        "technical expert (e.g., 'How does X actually work under the hood?', "
        "'What are the key engineering trade-offs?').\n\n"
        "5. FUTURIST QUESTIONS: 3-4 specific research questions for the "
        "future/culture expert (e.g., 'What happens if X becomes mainstream "
        "in 5 years?', 'Which industries get disrupted?').\n\n"
        "6. DEBATE ANGLES: 2-3 controversies, open questions, or opposing "
        "viewpoints that would make for interesting podcast discussion.\n\n"
        "7. TONE GUIDE: Should this episode be a serious deep-dive, a fun "
        "explainer, a debate, or something else? One sentence."
    ),
    expected_output=(
        "A structured research brief with all 7 numbered sections filled in. "
        "Each section should be concise but specific — the research agents "
        "need clear, actionable questions, not vague directions."
    ),
    agent=prompt_expander,
)

history_task = Task(
    description=(
        "Using the HISTORIAN QUESTIONS from the research brief, investigate the "
        "history and origins of the subject. Answer each question with specific "
        "facts, dates, names, and pivotal moments. Also note any debate angles "
        "from the brief that relate to history."
    ),
    expected_output="Detailed markdown bullet points answering each historian question from the brief.",
    agent=historian,
    context=[expansion_task],
    async_execution=True,
)

tech_task = Task(
    description=(
        "Using the TECHNOLOGIST QUESTIONS from the research brief, investigate "
        "the technical mechanics, engineering specs, and scientific principles "
        "of the subject. Answer each question with clear, accurate explanations. "
        "Also note any debate angles from the brief that relate to technology."
    ),
    expected_output="Detailed markdown bullet points answering each technologist question from the brief.",
    agent=technologist,
    context=[expansion_task],
    async_execution=True,
)
    
future_task = Task(
    description=(
        "Using the FUTURIST QUESTIONS from the research brief, investigate "
        "future trends, upcoming developments, and cultural impact of the "
        "subject. Answer each question with specific predictions and examples. "
        "Also note any debate angles from the brief that relate to the future."
    ),
    expected_output="Detailed markdown bullet points answering each futurist question from the brief.",
    agent=futurist,
    context=[expansion_task],
    async_execution=True,
)

rating_task = Task(
    description=(
        "Review the three research contributions (History, Technology, Future). "
        "For each one, assess:\n"
        "- Relevance: How well does it address the topic?\n"
        "- Depth: How detailed and informative is it?\n"
        "- Accuracy: Does it seem factually sound?\n\n"
        "Assign each a score from 0 to 100, then compute a weighting percentage "
        "that the scriptwriter should use to allocate dialogue time. "
        "The three percentages must sum to 100."
    ),
    expected_output=(
        "A rating for each research section with scores and a final weighting, e.g.:\n"
        "History: 85/100 → 45%\n"
        "Technology: 70/100 → 35%\n"
        "Future: 40/100 → 20%\n"
        "Plus a brief justification for each score."
    ),
    agent=critic,
    context=[history_task, tech_task, future_task],
)

writing_task = Task(
    description=(
        "Using ALL of the research notes, the critic's quality ratings, AND the "
        "original research brief (especially the AUDIENCE HOOK, DEBATE ANGLES, "
        "and TONE GUIDE), write a lively podcast script between Ali and Amir.\n\n"
        "Guidelines:\n"
        "- Open with the audience hook to grab the listener immediately.\n"
        "- Allocate dialogue proportionally to the critic's weighting — higher-rated "
        "research gets more airtime.\n"
        "- Weave in at least one debate angle where Ali and Amir have a playful "
        "disagreement or explore opposing viewpoints.\n"
        "- Match the tone suggested in the brief.\n"
        "- Each line must include a speaker ('Ali' or 'Amir'), text, and emotion.\n\n"
        "TARGET LENGTH: approximately {duration_minutes} minutes long. "
        "Assume ~150 spoken words per minute."
    ),
    expected_output=(
        "A complete podcast script in strict JSON format with a 'dialogue' list "
        "where each entry has 'speaker', 'text', and 'emotion' fields."
    ),
    agent=scriptwriter,
    context=[expansion_task, history_task, tech_task, future_task, rating_task],
    output_pydantic=PodcastScript,
)


# ── Orchestration ────────────────────────────────────────────────────────────

def generate_podcast_script(topic: str, duration_minutes: int = 5) -> str:
    for agent in [historian, technologist, futurist]:
        agent.tools = _research_tools
    return _run_crew(topic, duration_minutes)


def _run_crew(topic: str, duration_minutes: int) -> str:
    crew = Crew(
        agents=[prompt_expander, historian, technologist, futurist, critic, scriptwriter],
        tasks=[expansion_task, history_task, tech_task, future_task, rating_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={"topic": topic, "duration_minutes": duration_minutes})

    if result.pydantic:
        script_json = result.pydantic.model_dump_json(indent=2)
    else:
        script_json = str(result)

    # Save timestamped script to output/
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = output_dir / f"script_{timestamp}.json"
    script_path.write_text(script_json)
    logger.info("Script saved to %s", script_path)

    return script_json


if __name__ == "__main__":
    output = generate_podcast_script("Agentic AI", duration_minutes=5)
    print(output)
