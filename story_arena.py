import argparse
import asyncio
import logging
import os
from enum import StrEnum
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

TEMPERATURE = 0.3
LOG_PROMPTS = False
MODELS = {
    "rits:deepseek-v2-5": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v2-5",
        "model_name": "deepseek-ai/DeepSeek-V2.5",
    },
    "rits:granite-3-3-8b-instruct": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct",
        "model_name": "ibm-granite/granite-3.3-8b-instruct",
    },
    "rits:granite-4-0-small-franconia": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-0-small-franconia",
        "model_name": "ibm-granite/granite-4.0-small-prerelease-franconia.r250523a",
    },
    "rits:llama-4-mvk-17b-128e-fp8": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8",
        "model_name": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    },
    "rits:llama-4-scout-17b-16e": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-scout-17b-16e",
        "model_name": "meta-llama/llama-4-scout-17b-16e",
    },
    "rits:microsoft-phi-4": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4",
        "model_name": "microsoft/phi-4",
    },
    "rits:microsoft-phi-4-reasoning": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/phi-4-reasoning",
        "model_name": "microsoft/Phi-4-reasoning",
    },
    "rits:qwen2-5-72b-instruct": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct",
        "model_name": "Qwen/Qwen2.5-72B-Instruct",
    }
}
# DEFAULT_MODEL = 'rits:llama-4-mvk-17b-128e-fp8'
DEFAULT_MODEL = 'rits:granite-3-3-8b-instruct'

def log_prompt(components: list[AnyMessage]):
    if LOG_PROMPTS:
        console.print("[grey35] → Prompt[/]")
        for c in components:
            console.print(f"[grey35]{c}[/]")


def parse_model_name_from_position_statement(
    position_statement: str,
) -> tuple[str, str] | tuple[None, str]:
    """Parse the model name from the first line of the given position statement.
    For example:
    ```
      # rits:granite-3-3-8b-instruct
      My position is...
    ```

    results in `rits:granite-3-3-8b-instruct`"""
    lines = position_statement.split("\n")
    if lines[0].startswith("#"):
        model = lines[0].split("#")[1].strip()
        rest = "\n".join(lines[1:])
        if model in MODELS.keys():
            return (model, rest)
        else:
            return (None, rest)
    else:
        console.print('[red] no # found[/]')
        return (None, position_statement)


def create_llm(model_name: str, api_key="") -> ChatModel | OpenAIChatModel:
    if model_name.startswith("ollama:"):
        llm = ChatModel.from_name(
            model_name.split(':')[1], options=ChatModelParameters(temperature=TEMPERATURE)
        )
    elif model_name.startswith("rits:"):
        info = MODELS[model_name]
        settings = {
            "extra_headers": {"RITS_API_KEY": api_key},
            "api_key": api_key,
            "base_url": info["base_url"] + "/v1",
        }
        llm = OpenAIChatModel(
            model_id=info['model_name'],
            settings=settings,
            parameters=ChatModelParameters(temperature=TEMPERATURE),
        )
    else:
        raise ValueError(
            "Invalid MODEL_NAME. It should start with 'ollama:' or 'openai:'"
        )
    return llm


# Ideation phase for a single agent
async def run_ideation_phase(agent) -> str:
    ideation_prompt = dedent(
        text="""You are in the Ideation phase of the story-writing process.
             During this phase, your task is to brainstorm story ideas including
             genre, setting, characters, and shape. Ideate specific details
             for each of these story elements that will serve to support
             your position statement. Be specific with the details. For example,
             give your characters names, personalities, and backstories. Do not
             insert your own name or the names of your co-writers into the
             story."""
    )

    class Shape(StrEnum):
        rags_to_riches = "Rags to riches"
        tragedy = "Tragedy"
        man_in_hole = "Man in hole"
        icarus = "Icarus"
        cinderella = "Cinderella"
        oedipus = "Oedipus"
        which_way_is_up = "Which way is up?"
        creation_story = "Creation story"
        old_testament = "Old testament"
        new_testament = "New testament"
        other = "Other"

    class StoryShape(BaseModel):
        shape: Shape = Field(
            description=dedent(
                text="""The shape of the story. Story shapes are generally represented on a 2-D plane with the x-axis representing the beginning and end of a story and the y-axis representing represents good and bad fortune. Canonical story shapes are:
                - Rags to riches (rise): The main character starts off poorly but then their situation improves, either continually or dramatically.
                - Tragedy (fall): The main character starts off well but then their situation gets continually worse with no hope for improvement.
                - Man in Hole (fall-rise): The main character gets into trouble then gets out of it again and ends up better off for the experience.
                - Icarus (rise-fall): The main character's situation improves until a point at which it dramatically worsens.
                - Cinderella (rise-fall-rise): The main character comes across something wonderful, gets it, loses it, then gets it back forever.
                - Oedipus (fall-rise-fall): The main character loses something, gains it again, but then loses it again.
                - Which Way is Up?: The story has a lifelike ambiguity that keeps us from knowing if new developments are good or bad.
                - Creation Story: In many cultures' creation stories, humankind receives incremental gifts from a deity. Frist major staples like the earth and the sky, then smaller things like sparrows and cell phones. Not a common shape for Western stories, however.
                - Old Testament: Humankind receives incremental gifts from a deity, but is suddenly ousted from good standing in a fall of enormous proportions.
                - New Testament: Humankind receives incremental gifts from a deity, is suddenly ousted from good standing, but then receives off-the-charts bliss.

                Other shapes may also be possible.
                """
            )
        )
        description: str = Field(
            description="Narrative description of the story's shape"
        )

    class IdeationResult(BaseModel):
        # TODO: "Narrative shape" instead of conflict?
        # TODO: how to get the characters to use different forms of dialogue
        # TODO: "in media res"
        genre: str = Field(
            description="The genre of the story, such as whether it is a tragedy, comedy, satire, romance, etc."
        )
        setting: str = Field(
            description="The setting of the story. The setting includes specific details of the location or place, season, era, etc."
        )
        characters: str = Field(
            description="The characters who are involved in the story. Each character has a name, a backstory, personal details, and personality traits. Do not include your own name in the list of characters."
        )
        # conflict: str = Field(
        #     description="A description of a type of conflict between one or more of the characters"
        # )
        # sources:
        #  https://bigthink.com/high-culture/vonnegut-shapes/
        #  https://link.springer.com/content/pdf/10.1140/epjds/s13688-016-0093-1.pdf
        # shape: str = Field(
        #     description=dedent(
        #         text="""The shape of the story. Story shapes are generally represented on a 2-D plane with the x-axis representing the beginning and end of a story and the y-axis representing represents good and bad fortune. Canonical story shapes are:
        #         - Rags to riches (rise): The main character starts off poorly but then their situation improves, either continually or dramatically.
        #         - Tragedy (fall): The main character starts off well but then their situation gets continually worse with no hope for improvement.
        #         - Man in Hole (fall-rise): The main character gets into trouble then gets out of it again and ends up better off for the experience.
        #         - Icarus (rise-fall): The main character's situation improves until a point at which it dramatically worsens.
        #         - Cinderella (rise-fall-rise): The main character comes across something wonderful, gets it, loses it, then gets it back forever.
        #         - Oedipus (fall-rise-fall): The main character loses something, gains it again, but then loses it again.
        #         - Which Way is Up?: The story has a lifelike ambiguity that keeps us from knowing if new developments are good or bad.
        #         - Creation Story: In many cultures' creation stories, humankind receives incremental gifts from a deity. Frist major staples like the earth and the sky, then smaller things like sparrows and cell phones. Not a common shape for Western stories, however.
        #         - Old Testament: Humankind receives incremental gifts from a deity, but is suddenly ousted from good standing in a fall of enormous proportions.
        #         - New Testament: Humankind receives incremental gifts from a deity, is suddenly ousted from good standing, but then receives off-the-charts bliss.

        #         Other shapes may also be possible. Determine a shape that fits your position statement.
        #     """)
        # )
        shape: StoryShape = Field(description="The shape of the story")
        other_elements: str = Field(
            description="Other elements or details that are pertinent to the kind of story you want to tell"
        )

    log_prompt(
        components=[
            SystemMessage(agent["system_prompt"]),
            SystemMessage(ideation_prompt),
        ]
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
                    SystemMessage(ideation_prompt),
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
        Shape: {result.object["shape"]["shape"]}. {result.object["shape"]["description"]}
        Other elements: {result.object["other_elements"]}
        """
    )

    return ideation_result


class Utterance(BaseModel):
    message: str = Field(
        description=dedent(
            text="""A single utterance in a chat. The utterance does
                 not begin with the name of its author. It also does not contain
                 any utterances from any other writers involved in the chat."""
        )
    )


# Consensus phase with round-robin conversation
async def run_consensus_phase(
    agents, min_agent_turns=2, max_agent_turns=20
) -> list[UserMessage]:
    min_turns = min_agent_turns * len(agents)
    max_turns = max_agent_turns * len(agents)
    consensus_prompt = dedent(
        text="""You are in the Consensus phase of the story-writing process. The way this
              phase works is that you will have a conversation with the other writers in
              the writing room about the ideas you brainstormed in the ideation phase. The
              goal is to argue for your ideas on the details of the story (setting,
              characters, shape, etc.), while considering the arguments of the other
              writers. Here are the rules of the Consensus phase:

              - Share your own opinions before blindly agreeing to those of other writers.
              - It is important to argue for choices that align with and support
              your position statement.
              - You may address the other writers directly by name, but do not speak for them.
              - Do not prefix your messages with your own name or the names of other writers.
              - Do not repeat statements that have already been made.
              - If you have nothing to say, say \"I have nothing else to add.\"
              """
    )
    consensus_prompt_yield = dedent(
        text="""- When you are satisfied with all of the decisions that have been made, tell the
             group "I yield" (and only "I yield" with no other text). If the group does not
             reach consensus after each writer takes {max_agent_turns} turns ({max_turns}
             turns overall), the Consensus phase will end, possibly with some decisions left
             unmade."""
    )

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

                # enforce a minimum number of conversational turns before the writers
                # are given instructions that they can yield
                if turn <= min_turns:
                    prompt = consensus_prompt
                else:
                    prompt = consensus_prompt + consensus_prompt_yield

                prompt_components = [
                    SystemMessage(content=agent["system_prompt"]),
                    SystemMessage(content=prompt),
                    *conversation,
                ]

                log_prompt(components=prompt_components)
                result = await agent["llm"].create_structure(
                    schema=Utterance,
                    messages=prompt_components,
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
                 of your interpretation of the major decisions that were made.
                 Your interpretation may differ from other writers. You may also
                 may wish to override the consensus decisions that were reached
                 if you believe that your own views will ultimately result in a
                 better story. You will use this summary to aid your writing
                 process."""
        )

        console.print(f"[cyan][bold]{agent['name']}'s summary[/bold][/]")
        result = await agent["llm"].create(
            messages=[
                SystemMessage(agent["system_prompt"]),
                SystemMessage(summary_prompt),
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
    sentence: str = Field(description="A *single* sentence in a story.")
    author: str = Field(description="The author of the sentence.")

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
        text=f"""You are in the Writing phase of the story-writing process. Your job is
              to work with the other writers in the room to construct a story based on
              your own understanding of the consensus reached during the consensus
              phase. You will work in a round-robin fashion with your fellow writers
              by contributing one sentence to the story at a time. Here are the rules
              for the Writing phase:
              - Stories have a clear beginning, middle, and end. Make sure you and
              your authors follow this overall structure. Do not begin your story
              in the middle!
              - Keep the story coherent, but also ensure it meets your own goals and
              understanding of what the story is supposed to be about.
              - You will have {num_iterations} turns at writing. Use them well.
              - Pay attention to the shape of the story and make sure that what you
              are writing conforms to the desired shape, given the current iteration
              you are on.
              - Do not insert yourself or your co-writers into the story.
              - Make sure your characters have actual names (not just descriptions).
              - Use dialogue! Make the characters speak.
              - Dialogue can come in many forms, including indirect, direct, and
              summarized. Direct dialogue involves directly quoting things a character
              says. It adds subtext by mapping out the characters' thoughts, emotions,
              physical setting, and actions without specifically mentioning those
              attributes. Indirect dialogue is where the narrator paraphrases the
              dialogue. Summarized dialogue condenses entire exchanges. Both indirect
              and summarized dialogue compress time, reveal character, and draw
              attention to direct dialogue.
              - When writing direct dialogue, use double quotation marks (") instead
              of single quotation marks (').
              - Do not repeat past sentences that already exist in the story.
              - Do not rehash past ideas that already exist in the story.
              - Do not insert newlines.
              - Vary the composition of the sentences in the story. Try not to start
              each sentence with the same words used at the start of previous
              sentences (e.g. don't keep starting sentences with "As").
              - Write a mixture of short and long sentences.
              - Include sentences that are purely dialogue! Not every sentence needs
              to include a mix of narration and dialogue.
              - Find opportunities for parallel construction, which refers to using
              the same grammatical structure for two or more related elements in a
              sentence.
              - If you have nothing to add, do not add anything!"""
    )

    # write!
    story: Story = Story()
    for i in range(0, num_iterations):
        # invoke each agent for a turn at the pen
        for agent in agents:
            consensus_summary = agent["consensus_summary"]
            story_prompt = dedent(
                text=f"""
                Below is your understanding of the agreements made during the
                consensus phase. Use this to guide how you write the story.

                ###
                {consensus_summary}
                ###

                It is your turn to add a new sentence to the story. You are on
                iteration {i} of {num_iterations} ({(i / num_iterations) * 100}%
                of the way through the story). Below is the story so far. If you
                have nothing to contribute to the story, remain silent and do not
                add a new sentence to it.

                ###
                {story.render()}
                ###
                """
            )

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
                            SystemMessage(writing_prompt),
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
             makes it enticing for the reader to want to read the story."""
    )

    class Title(BaseModel):
        title: str = Field(description="The title of the story")

    result = await llm.create_structure(
        schema=Title,
        messages=[
            SystemMessage(title_prompt),
            UserMessage(f"This is the story:\n###\n{story.render()}\n###"),
        ],
        max_retries=5,
        abort_signal=AbortSignal.timeout(duration=20),
    )

    title = result.object["title"]
    return title


async def main() -> None:
    load_dotenv()
    RITS_API_KEY = os.getenv("RITS_API_KEY")

    # RITS_API_URL = os.getenv(
    #     key="RITS_API_URL",
    #     default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com",
    # )
    # MODEL_NAME = os.getenv("MODEL_NAME")

    if RITS_API_KEY is None:
        raise ValueError("RITS_API_KEY must be defined in the .env file.")

    parser = argparse.ArgumentParser(description="✏️ Story Arena!")
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

    # Read story prompt from file
    with open(args.story_prompt, "r") as file:
        story_prompt = file.read().strip()

    # Read prompt files and create agents
    agents = []
    for file_path in args.prompt_files:
        with open(file_path, "r") as file:
            name = file_path.split("/")[-1].split(".")[0]
            position_statement = file.read().strip()
            (model_name, position_statement) = parse_model_name_from_position_statement(
                position_statement=position_statement
            )
            if model_name is None:
                model_name = DEFAULT_MODEL
            llm = create_llm(model_name=model_name, api_key=RITS_API_KEY)
            agents.append(
                {
                    "name": name,
                    "position_statement": position_statement,
                    "llm": llm,
                    "model_name": model_name,
                }
            )

    all_agent_names = set([a["name"] for a in agents])
    for agent in agents:
        other_agent_names = all_agent_names
        other_agent_names.discard(agent["name"])
        prompt = dedent(
            text=f"""You are {agent["name"]} and you are in a
                  writer's room with {", ".join(other_agent_names)}.
                  Your task is to write a story together on this
                  topic: {story_prompt} Your position statement is
                  below.

                  ###
                  {agent["position_statement"]}
                  ###
                  """
        )
        agent["system_prompt"] = prompt

    # Print agent information
    console.print("[bold cyan]Welcome to Story Arena![/]")
    console.print(f"[grey35]Temperature: {TEMPERATURE}[/]")
    console.print(f"[grey35]Story prompt: {story_prompt}[/]")
    for agent in agents:
        console.print(
            f"[cyan]Writer {agent['name']} has entered the arena![/] [grey35]→ {agent['model_name']}[/]"
        )

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
    story = await run_writing_phase(agents=agents, num_iterations=8)
    title = await generate_story_title(llm=llm, story=story)

    console.print(f"[bold cyan]{'*' * 72}[/]")
    console.print(f"[bold cyan]{title}[/]")
    console.print(f"[cyan]by {', '.join(agent['name'] for agent in agents)}[/]")
    console.print()
    console.print(f"[cyan]{story.render()}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
