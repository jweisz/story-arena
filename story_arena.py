import argparse
import asyncio
import logging
import os
from textwrap import dedent

from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.backend import (
    AnyMessage,
    ChatModelParameters,
    SystemMessage,
    UserMessage,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import AbortError
from beeai_framework.utils.cancellation import AbortSignal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console

# Rich console
console = Console()

# Enable logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger(name="httpx").setLevel(level=logging.WARNING)
logging.getLogger(name="httpcore.connection").setLevel(level=logging.WARNING)
logging.getLogger(name="httpcore.http11").setLevel(level=logging.WARNING)
logging.getLogger(name="openai._base_client").setLevel(level=logging.WARNING)
logging.getLogger(name="asyncio").setLevel(level=logging.WARNING)

TEMPERATURE = 0.7
LOG_PROMPTS = False


def log_prompt(components: list[AnyMessage]):
    if LOG_PROMPTS:
        console.print("[grey35] → Prompt[/]")
        for c in components:
            console.print(f"[grey35]{c}[/]")


# Ideation phase for a single agent
async def run_ideation_phase(agent) -> str:
    ideation_prompt = dedent(
        text="""You are in the Ideation phase of the story-writing process.
             During this phase, your task is to brainstorm story ideas including
             genre, setting, characters, and conflict. Ideate specific details
             for each of these story elements. Make sure that your ideas conform
             to your position statement. Give each character a name, but do not
             insert your own name or the names of your co-authors into the
             story."""
    )

    class IdeationResult(BaseModel):
        genre: str = Field(
            description="The genre of the story, such as whether it is a tragedy, comedy, satire, romance, etc."
        )
        setting: str = Field(
            description="The setting of the story, with specific details of the location/place, the season, the era, etc."
        )
        characters: str = Field(
            description="A cast of characters who are involved in the story. Each character has a name, backstory, personal details, and personality traits. Do not include your own name in the list of characters."
        )
        conflict: str = Field(
            description="A description of a type of conflict between one or more of the characters"
        )
        other_elements: str = Field(
            description="Other elements that are pertinent to the kind of story you want to tell"
        )

    log_prompt(
        components=[SystemMessage(agent["system_prompt"]), UserMessage(ideation_prompt)]
    )

    success = False
    attempt = 0
    result = None
    while not success:
        try:
            result = await agent["llm"].create_structure(
                schema=IdeationResult,
                messages=[
                    SystemMessage(agent["system_prompt"]),
                    UserMessage(ideation_prompt),
                ],
                max_retries=3,
                abort_signal=AbortSignal.timeout(duration=30),
            )
            success = True
        except AbortError:
            attempt = attempt + 1
            console.print(
                f"[red bold]→ Writer {agent['name']} timed out, trying again (attempt {attempt})[/]"
            )
            # logging.getLogger(name='openai._base_client').setLevel(level=logging.DEBUG)

    if result is None:
        raise AbortError("Ideation phase failed to complete within the allotted time.")

    ideation_result = dedent(
        text=f"""These are your ideas for the story:
        Genre: {result.object["genre"]}
        Setting: {result.object["setting"]}
        Characters: {result.object["characters"]}
        Conflict: {result.object["conflict"]}
        Other elements: {result.object["other_elements"]}
        """)

    return ideation_result

class Utterance(BaseModel):
    message: str = Field(
        description=dedent(
            text="""A single utterance in a chat. The utterance does not contain
                 the name of the author. It also does not contain any utterances
                 from anyone else in the chat.""")
    )

# Consensus phase with round-robin conversation
async def run_consensus_phase(agents, num_agent_turns=20) -> list[UserMessage]:
    max_turns = num_agent_turns * len(agents)
    consensus_prompt = dedent(
        text=f"""You are in the consensus phase of the story-writing process. The way this
              phase works is that you will have a conversation with the other writers in
              the writing room about the ideas you brainstormed in the ideation phase. The
              goal is to argue for your ideas on the details of the story (setting,
              characters, conflict, etc.), while listening to the arguments of the other
              writers. It is important to argue for choices that align with and support
              your position statement. Do not make statements that indicate you are
              speaking for your fellow writers. Do not prefix your messages with your own
              name or the names of other writers. When you are satisfied with all of the
              decisions that have been made, tell the group "I yield" (and only "I
              yield"). If the group does not reach consensus after {max_turns} turns (you
              will get {num_agent_turns} turns), the consensus phase will end, possibly
              with some decisions left unmade. Do your best to reach consensus before this
              occurs.""")

    active_agents = set(agent["name"] for agent in agents)
    turn = 0

    # create the conversational space
    conversation: list[UserMessage] = []

    while len(active_agents) > 1:
        # give each agent a turn
        for agent in agents:
            # make sure they are still active
            if agent["name"] in active_agents:
                turn = turn + 1

                console.print(f"[cyan][bold][{turn}] {agent['name']}'s turn[/bold][/]")
                log_prompt(
                    components=[
                        SystemMessage(agent["system_prompt"]),
                        UserMessage(consensus_prompt),
                        *conversation,
                    ]
                )
                result = await agent["llm"].create_structure(
                    schema=Utterance,
                    messages=[
                        SystemMessage(agent["system_prompt"]),
                        UserMessage(consensus_prompt),
                        *conversation,
                    ],
                    max_retries=3,
                    abort_signal=AbortSignal.timeout(duration=30),
                )

                message = result.object["message"]
                conversation.append(UserMessage(f"{agent['name']}: {message}"))
                console.print(
                    f"[magenta][bold]{agent['name']}[/bold]: [cyan]|[/cyan]{message}[cyan]|[/cyan][/]"
                )

                if "i yield" in message.lower():
                    console.print(f"[green bold]Writer {agent['name']} has yielded[/]")
                    active_agents.remove(agent["name"])

                if turn > max_turns:
                    console.print(
                        f"[green bold]Turn number ({turn}) exceeds turn limit ({max_turns}), ending conversation[/]"
                    )
                    active_agents.clear()

    console.print("[green bold]All writers have yielded, moving to summarization[/]")

    # each agent summarizes the decisions made in the conversation
    for agent in agents:
        summary_prompt = dedent(
            text="""This is the conversation you just had
                 with the other writers in the writing room. Produce a summary
                 of the major decisions that were made, keeping in mind that you
                 may wish to override the consensus decisions that were reached
                 if you believe that your own views will result in a better
                 story. You will use this summary to aid your writing
                 process.""")

        console.print(f"[cyan][bold]{agent['name']}'s summary[/bold][/]")
        result = await agent["llm"].create(
            messages=[
                SystemMessage(agent["system_prompt"]),
                UserMessage(summary_prompt),
                *conversation,
            ],
            max_retries=3,
            abort_signal=AbortSignal.timeout(duration=30),
        )

        summary = result.messages[-1].content[-1].text
        console.print(f"[magenta][bold]{agent['name']}[/bold]: {summary}[/]")
        agent["consensus_summary"] = summary
    return conversation


# Writing phase with round-robin writing turns
class Sentence(BaseModel):
    sentence: str = Field(description="A sentence in a story.")
    author: str = Field(description="The author (or authors) of the sentence.")

    def __str__(self) -> str:
        return f"{self.sentence} [{self.author}]"


class Story:
    story: list[Sentence]

    def __init__(self) -> None:
        self.story = []

    def __str__(self) -> str:
        return "\n".join(str(sentence) for sentence in self.story)

    def render(self) -> str:
        return " ".join(sentence.sentence for sentence in self.story)


async def run_writing_phase(agents, num_iterations=20) -> Story:
    writing_prompt = dedent(
        text=f"""You are in the writing phase of the story-writing process. Your
              job is to work with the other writers in the room to construct a
              story based on your own understanding of the consensus reached
              during the consensus phase. You will work in a round-robin fashion
              with your fellow writers by contributing one sentence to the story
              at a time. Keep the story coherent, but also ensure it meets your
              own goals and understanding of what the story is supposed to be
              about. You will have {num_iterations} turns at writing. Use them
              well. Remember that a story has three parts: the introduction, the
              body, and the conclusion. Work together to ensure the story
              contains these three parts and is coherent. Do not insert yourself
              or your co-writers into the story.""")

    # write!
    story: Story = Story()
    for i in range(0, num_iterations):
        # invoke each agent for a turn at the pen
        for agent in agents:
            consensus_summary = agent['consensus_summary']
            story_prompt = dedent(
                text=f"""
                This is your understanding of the agreements made during the
                consensus phase. Use this to guide how you write the story.
                {consensus_summary}

                It is your turn to add a new sentence to the story. You are on
                iteration {i} of {num_iterations}. This is the story so far. If
                you have nothing to contribute to the story, remain silent and
                do not add a new sentence to it.
                {story.render()}""")

            # for whatever reason, RITS likes to timeout on these calls, so we need to be a bit more
            # agressive...
            success = False
            attempt = 0
            result = None
            while not success:
                try:
                    result = await agent["llm"].create_structure(
                        schema=Sentence,
                        messages=[
                            SystemMessage(agent["system_prompt"]),
                            UserMessage(writing_prompt),
                            UserMessage(story_prompt),
                        ],
                        max_retries=5,
                        abort_signal=AbortSignal.timeout(duration=20),
                    )
                    success = True
                except AbortError:
                    attempt = attempt + 1
                    console.print(
                        f"[red bold]→ Writer {agent['name']} timed out, trying again (attempt {attempt})[/]"
                    )
                    # logging.getLogger(name='openai._base_client').setLevel(level=logging.DEBUG)

            # TODO maybe pass on this writer's turn if they time out too much?
            if result is None:
                raise AbortError(
                    "Writing phase failed to complete within the allotted time."
                )

            new_sentence = result.object["sentence"]

            if new_sentence is None or new_sentence == "":
                console.print(f"[green bold]Writer {agent['name']} has yielded[/]")
            else:
                console.print(
                    f"[green bold][{i}][/green bold] [magenta][bold][{agent['name']}][/bold]: {new_sentence}[/]"
                )
                story.story.append(
                    Sentence(sentence=new_sentence, author=agent["name"])
                )
    return story

async def generate_story_title(llm, story: Story) -> str:
    title_prompt = dedent(
        text="""You are a publisher. Your job is to read a story and generate a
             compelling title for it that captures the essence of the story and
             makes it enticing for the reader to want to read the story.""")

    class Title(BaseModel):
        title: str = Field(description="The title of the story")

    result = await llm.create_structure(
        schema=Title,
        messages=[
            SystemMessage(title_prompt),
            UserMessage(f"This is the story:\n{story.render()}")
        ],
        max_retries=5,
        abort_signal=AbortSignal.timeout(duration=20),
    )

    title = result.object['title']
    return title


async def main() -> None:
    load_dotenv()
    RITS_API_KEY = os.getenv("RITS_API_KEY")
    RITS_API_URL = os.getenv(
        key="RITS_API_URL",
        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com",
    )
    MODEL_NAME = os.getenv("MODEL_NAME")

    if RITS_API_KEY is None or RITS_API_URL is None or MODEL_NAME is None:
        raise ValueError(
            "RITS_API_KEY, RITS_API_URL, and MODEL_NAME must be defined in the .env file."
        )

    parser = argparse.ArgumentParser(description="Multi-Agent Story Writer")
    parser.add_argument(
        "prompt_files", type=str, nargs="+", help="Paths to agent prompt files"
    )
    parser.add_argument(
        "--story-prompt",
        type=str,
        required=True,
        help="Path to file containing the story prompt",
    )
    args = parser.parse_args()

    # Initialize LLM
    if MODEL_NAME.startswith("ollama:"):
        llm = ChatModel.from_name(
            MODEL_NAME, options=ChatModelParameters(temperature=TEMPERATURE)
        )
    elif MODEL_NAME.startswith("openai:"):
        settings = {
            "extra_headers": {"RITS_API_KEY": RITS_API_KEY},
            "api_key": RITS_API_KEY,
            "base_url": RITS_API_URL,
        }
        llm = OpenAIChatModel(
            model_id=MODEL_NAME.split(":")[1],
            settings=settings,
            parameters=ChatModelParameters(temperature=TEMPERATURE),
        )
    else:
        raise ValueError(
            "Invalid MODEL_NAME. It should start with 'ollama:' or 'openai:'"
        )

    # Read story prompt from file
    with open(args.story_prompt, "r") as file:
        story_prompt = file.read().strip()

    # Read prompt files and create agents
    agents = []
    for file_path in args.prompt_files:
        with open(file_path, "r") as file:
            name = file_path.split("/")[-1].split(".")[0]
            position_statement = file.read().strip()
            agents.append(
                {"name": name, "position_statement": position_statement, "llm": llm}
            )

    all_agent_names = set([a["name"] for a in agents])
    for agent in agents:
        other_agent_names = all_agent_names
        other_agent_names.discard(agent["name"])
        prompt = dedent(
            text=f"""You are {agent["name"]} and you are in a
                  writer's room with {", ".join(other_agent_names)}.
                  Your task is to write a story together on this
                  topic: {story_prompt} Your position statement is:
                  {agent["position_statement"]}""")
        agent["system_prompt"] = prompt

    # Print agent information
    console.print("[bold cyan]Welcome to Story Arena![/]")
    console.print(f"[grey35]LLM: {MODEL_NAME}[/]")
    console.print(f"[grey35]Temperature: {TEMPERATURE}[/]")
    console.print(f"[grey35]Story prompt: {story_prompt}[/]")
    for agent in agents:
        console.print(f"[cyan]Writer {agent['name']} has entered the arena![/]")

    # Run the workflow
    console.print(f"[bold cyan]{'*' * 72}[/]")
    console.print("[bold cyan]Starting Ideation Phase...[/]")
    for agent in agents:
        console.print(f"[cyan][bold]Ideation[/bold]: {agent['name']}'s turn[/]")
        response = await run_ideation_phase(agent=agent)
        console.print(f"[magenta][bold]{agent['name']}[/bold]: {response}[/]")
        agent["ideation_result"] = response

    console.print(f"[bold cyan]{'*' * 72}[/]")
    console.print("[bold cyan]Starting Consensus Phase...[/]")
    _ = await run_consensus_phase(agents=agents)

    console.print(f"[bold cyan]{'*' * 72}[/]")
    console.print("[bold cyan]Starting Writing Phase...[/]")
    story = await run_writing_phase(agents=agents, num_iterations=10)
    title = await generate_story_title(llm=llm, story=story)

    console.print(f"[bold cyan]{'*' * 72}[/]")
    console.print(f"[bold cyan]{title}[/]")
    console.print(f"[cyan]by {', '.join(agent['name'] for agent in agents)}[/]")
    console.print()
    console.print(f"[cyan]{story.render()}[/cyan]")

if __name__ == "__main__":
    asyncio.run(main())
