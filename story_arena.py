import argparse
import asyncio
import datetime
import logging
import os
import random
from textwrap import dedent

from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
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

# Global constants
TIMEOUT_DURATION = 120

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
    "rits:mistral-small-3-1-24b-2503": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-small-3-1-24b-2503",
        "model_name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    },
    "rits:qwen2-5-72b-instruct": {
        "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct",
        "model_name": "Qwen/Qwen2.5-72B-Instruct",
    },
    "watsonx:granite-3-3-8b-instruct": {
        "model_name": "ibm/granite-3-3-8b-instruct",
    },
    "watsonx:llama-4-maverick-17b-128e-instruct-fp8": {
        "model_name": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    },
    "watsonx:mistral-large": {
        "model_name": "mistralai/mistral-large"
    },
    "watsonx:mistral-small-3-1-24b-instruct-2503": {
        "model_name": "mistralai/mistral-small-3-1-24b-instruct-2503"
    },
}

def setup_logging(log_dir=None) -> None:
    def create_log_directory() -> str:
        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        log_dir = f"runs/run-{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    if log_dir is None:
        log_dir = create_log_directory()

    console.print(f"[grey35]Logging to {log_dir}[/]")

    # Send logs to a file
    logger = logging.getLogger(name="story_arena")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=os.path.join(log_dir, "console.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Send prompt logs to a file
    prompt_logger = logging.getLogger(name="story_arena_prompts")
    prompt_logger.setLevel(logging.INFO)
    prompt_handler = logging.FileHandler(filename=os.path.join(log_dir, "prompts.log"))
    prompt_handler.setLevel(level=logging.INFO)
    prompt_logger.addHandler(prompt_handler)

    # Log debugging info from these libraries to a file
    debug_file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, "debug.log")
    )
    debug_file_handler.setLevel(level=logging.DEBUG)
    for logger_name in [
        "httpx",
        "httpcore.connection",
        "httpcore.http11",
        "openai._base_client",
        "asyncio",
    ]:
        logging.getLogger(name=logger_name).setLevel(logging.DEBUG)
        logging.getLogger(name=logger_name).addHandler(debug_file_handler)
        formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
        debug_file_handler.setFormatter(formatter)


def log(msg: str) -> None:
    console.print(msg)
    logging.getLogger(name="story_arena").info(msg=msg)


def log_prompt(components: list[AnyMessage]) -> None:
    logger = logging.getLogger(name="story_arena_prompts")
    logger.info(msg=f"{'*' * 27} → Prompt ← {'*' * 27}")
    for c in components:
        logger.info(f"{c}")


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
        return (model, rest)
    else:
        console.print("[red] no # found[/]")
        return (None, position_statement)


def create_llm(
    model_name: str, temperature: float, api_key=""
) -> ChatModel | OpenAIChatModel:
    load_dotenv()

    if model_name.startswith("ollama:"):
        llm = ChatModel.from_name(
            model_name,
            options={
                'temperature': temperature
            },
        )
    elif model_name.startswith("rits:"):
        api_key = os.getenv("RITS_API_KEY")
        info = MODELS[model_name]
        settings = {
            "extra_headers": {"RITS_API_KEY": api_key},
            "api_key": api_key,
            "base_url": info["base_url"] + "/v1",
        }
        llm = OpenAIChatModel(
            model_id=info["model_name"],
            settings=settings,
            parameters=ChatModelParameters(temperature=temperature),
        )
    elif model_name.startswith("watsonx"):
        base_url = os.getenv("WATSONX_BASE_URL")
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        info = MODELS[model_name]
        settings = {
            "base_url": base_url,
            "api_key": api_key,
            "project_id": project_id,
        }
        llm = WatsonxChatModel(model_id=info["model_name"], settings=settings)
    else:
        raise ValueError(
            "Invalid MODEL_NAME. It should start with 'ollama:' or 'openai:'"
        )
    return llm


# Ideation phase for a single agent
async def run_ideation_phase(agent) -> str:
    ideation_prompt = dedent(
        text="""
        You are in the Ideation phase of the story-writing process. During this
        phase, your task is to brainstorm story ideas including genre, setting,
        characters, shape, plot, and conflict. Here are the rules to follow for
        this phase:

        - Ideate specific details for each of the story elements. These details
          should support your position statement.
        - Be specific with the details. For example, give your characters names,
          personalities, and backstories. For example, a character might be
          audacious, meek, curious, or confrontational. They might not respect
          authority. They might have had a bad childhood experience with AI. Get
          creative!
        - Do not insert your own name or the names of your co-writers into the
          story."""
    )

    # class Shape(StrEnum):
    #     rags_to_riches = "Rags to riches"
    #     tragedy = "Tragedy"
    #     man_in_hole = "Man in hole"
    #     icarus = "Icarus"
    #     cinderella = "Cinderella"
    #     oedipus = "Oedipus"
    #     which_way_is_up = "Which way is up?"
    #     creation_story = "Creation story"
    #     old_testament = "Old testament"
    #     new_testament = "New testament"
    #     other = "Other"

    # class StoryShape(BaseModel):
    #     shape: Shape = Field(
    #         description=dedent(
    #             text="""The shape of the story. Story shapes are generally represented on a 2-D plane with the x-axis representing the beginning and end of a story and the y-axis representing represents good and bad fortune. Canonical story shapes are:
    #             - Rags to riches (rise): The main character starts off poorly but then their situation improves, either continually or dramatically.
    #             - Tragedy (fall): The main character starts off well but then their situation gets continually worse with no hope for improvement.
    #             - Man in Hole (fall-rise): The main character gets into trouble then gets out of it again and ends up better off for the experience.
    #             - Icarus (rise-fall): The main character's situation improves until a point at which it dramatically worsens.
    #             - Cinderella (rise-fall-rise): The main character comes across something wonderful, gets it, loses it, then gets it back forever.
    #             - Oedipus (fall-rise-fall): The main character loses something, gains it again, but then loses it again.
    #             - Which Way is Up?: The story has a lifelike ambiguity that keeps us from knowing if new developments are good or bad.
    #             - Creation Story: In many cultures' creation stories, humankind receives incremental gifts from a deity. Frist major staples like the earth and the sky, then smaller things like sparrows and cell phones. Not a common shape for Western stories, however.
    #             - Old Testament: Humankind receives incremental gifts from a deity, but is suddenly ousted from good standing in a fall of enormous proportions.
    #             - New Testament: Humankind receives incremental gifts from a deity, is suddenly ousted from good standing, but then receives off-the-charts bliss.

    #             Other shapes may also be possible.
    #             """
    #         )
    #     )
    #     description: str = Field(
    #         description="Narrative description of the story's shape"
    #     )

    class IdeationResult(BaseModel):
        genre: str = Field(
            description="The genre of the story, such as whether it is a tragedy, comedy, satire, romance, etc."
        )
        setting: str = Field(
            description="The setting of the story. The setting is a specific location or place. It may also involve a specific season or time of year or a specific era or epoch."
        )
        characters: str = Field(
            description="The characters who are involved in the story. Each character has a name, a backstory, personal details, and personality traits. Do not include your own name in the list of characters."
        )
        # sources:
        #  https://bigthink.com/high-culture/vonnegut-shapes/
        #  https://link.springer.com/content/pdf/10.1140/epjds/s13688-016-0093-1.pdf
        shape: str = Field(
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

                Other shapes may also be possible. Determine a shape that fits your position statement.
            """
            )
        )
        # shape: StoryShape = Field(description="The shape of the story")
        plot: str = Field(
            description="The sequence of events that occurs in the story from the beginning to the middle to the end."
        )
        conflict: str = Field(
            description="A description of a type of conflict between one or more of the characters, or a conflict within one single character"
        )
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
                abort_signal=AbortSignal.timeout(duration=TIMEOUT_DURATION),
            )
            success = True
        except AbortError:
            attempt = attempt + 1
            log(
                f"[red bold]→ Writer {agent['name']} timed out, trying again (attempt {attempt})[/]"
            )
            # logging.getLogger(name='openai._base_client').setLevel(level=logging.DEBUG)

    if result is None:
        raise AbortError("Ideation phase failed to complete within the allotted time.")

    try:
        ideation_result = dedent(
            text=f"""These are your ideas for the story:
            Genre: {result.object.get("genre", "Unknown")}
            Setting: {result.object.get("setting", "Unknown")}
            Characters: {result.object.get("characters", "Unknown")}
            Shape: {result.object.get("shape", "Unknown")}
            Plot: {result.object.get("plot", "Unknown")}
            Conflict: {result.object.get("conflict", "Unknown")}
            Other elements: {result.object.get("other_elements", "Unknown")}
            """
        )
    except KeyError as e:
        log(f"[red bold]Unable to deserialize LLM response:\n{repr(result)}")
        raise e

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
    agents, min_agent_turns=2, max_agent_turns=10
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
              - Challenge the ideas of other writers before accepting them.
              - Argue for choices that align with and support your position statement.
              - You may address the other writers directly by name, but do not speak for them.
              - Do not prefix your messages with your own name or the names of other writers.
              - Do not repeat statements that have already been made.
              - Do not summarize or consolidate the decisions made in the conversation.
              - Speak in short sentences, not in paragraphs. Phrase things the way you would
                if you were speaking out loud.
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

    # randomize the order
    shuffled_agents = agents
    random.shuffle(shuffled_agents)

    while len(active_agents) > 1:
        # give each agent a turn
        for agent in shuffled_agents:
            # make sure they are still active
            if agent["name"] in active_agents:
                turn = turn + 1

                log(f"[cyan][bold][{turn}] {agent['name']}'s turn[/bold][/]")

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

                success = False
                attempt = 0
                result = None
                while not success:
                    try:
                        result = await agent["llm"].create_structure(
                            schema=Utterance,
                            messages=prompt_components,
                            max_retries=3,
                            abort_signal=AbortSignal.timeout(duration=TIMEOUT_DURATION),
                        )
                        success = True
                    except AbortError:
                        attempt = attempt + 1
                        console.print(
                            f"[red bold]→ Writer {agent['name']} timed out, trying again (attempt {attempt})[/]"
                        )

                if result is None:
                    raise AbortError(
                        "Consensus phase failed to complete within the allotted time."
                    )

                message = result.object.get("message", "")
                conversation.append(UserMessage(f"{agent['name']}: {message}"))
                log(
                    f"[magenta][bold]{agent['name']}[/bold]: [cyan]|[/cyan]{message}[cyan]|[/cyan][/]"
                )

                if "i yield" in message.lower():
                    log(f"[green bold]Writer {agent['name']} has yielded[/]")
                    active_agents.remove(agent["name"])

                if turn > max_turns:
                    log(
                        f"[green bold]Turn number ({turn}) exceeds turn limit ({max_turns}), ending conversation[/]"
                    )
                    active_agents.clear()

    log("[green bold]All writers have yielded, moving to summarization[/]")

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

        log(f"[cyan][bold]{agent['name']}'s summary[/bold][/]")

        success = False
        attempt = 0
        result = None
        while not success:
            try:
                result = await agent["llm"].create(
                    messages=[
                        SystemMessage(agent["system_prompt"]),
                        SystemMessage(summary_prompt),
                        *conversation,
                    ],
                    max_retries=3,
                    abort_signal=AbortSignal.timeout(duration=TIMEOUT_DURATION),
                )
                success = True
            except AbortError:
                attempt = attempt + 1
                log(
                    f"[red bold]→ Writer {agent['name']} timed out, trying again (attempt {attempt})[/]"
                )

        if result is None:
            raise AbortError("Summary failed to complete within the allotted time.")

        summary = result.messages[-1].content[-1].text
        log(f"[magenta][bold]{agent['name']}[/bold]: {summary}[/]")
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


async def run_writing_phase(agents, style_instructions: str, num_iterations: int=10) -> Story:
    writing_prompt = dedent(
        text=f"""
        You are in the Writing phase of the story-writing process. Your job is
        to work with the other writers in the room to construct a story based on
        a discussion you had with the other writers. You will work in a
        round-robin fashion with your fellow writers by contributing one
        sentence to the story at a time. Here are the rules for the Writing
        phase:
        - You will have {num_iterations} turns at writing. Use them well.
        - When quoting dialogue, use double quotation marks (") instead
          of single quotation marks (').
        - Do not repeat or rephrase past sentences that already exist in the story.
        - Do not rehash past ideas that already exist in the story.
        - Keep a consistent perspective and tell the story with a consistent
          narrator.
        - DO NOT REPEAT SENTENCES THAT ALREADY EXIST!
        """
    )

    if style_instructions is not None:
        writing_prompt = writing_prompt + f"""- {style_instructions}"""

    # text=f"""You are in the Writing phase of the story-writing process. Your job is
    #       to work with the other writers in the room to construct a story based on
    #       a discussion you had with the other writers. You will work in a round-robin
    #       fashion with your fellow writers by contributing one sentence to the story
    #       at a time. Here are the rules for the Writing phase:

    #       - Stories have a clear beginning, middle, and end. Make sure you and
    #       your authors follow this overall structure. Do not begin your story
    #       in the middle!
    #       - Keep the story coherent, but also ensure it meets your own goals and
    #       understanding of what the story is supposed to be about.
    #       - You will have {num_iterations} turns at writing. Use them well.
    #       - Pay attention to the shape of the story and make sure that what you
    #       are writing conforms to the desired shape, given the current iteration
    #       you are on.
    #       - Do not insert yourself or your co-writers into the story.
    #       - Make sure your characters have actual names (not just descriptions).
    #       - Use dialogue! Make the characters speak.
    #       - Dialogue can come in many forms, including indirect, direct, and
    #       summarized. Direct dialogue involves directly quoting things a character
    #       says. It adds subtext by mapping out the characters' thoughts, emotions,
    #       physical setting, and actions without specifically mentioning those
    #       attributes. Indirect dialogue is where the narrator paraphrases the
    #       dialogue. Summarized dialogue condenses entire exchanges. Both indirect
    #       and summarized dialogue compress time, reveal character, and draw
    #       attention to direct dialogue.
    #       - When writing direct dialogue, use double quotation marks (") instead
    #       of single quotation marks (').
    #       - Do not repeat past sentences that already exist in the story.
    #       - Do not rehash past ideas that already exist in the story.
    #       - Do not insert newlines.
    #       - Vary the composition of the sentences in the story. Try not to start
    #       each sentence with the same words used at the start of previous
    #       sentences (e.g. don't keep starting sentences with "As").
    #       - Write a mixture of short and long sentences.
    #       - Include sentences that are purely dialogue! Not every sentence needs
    #       to include narration.
    #       - Find opportunities for parallel construction, which refers to using
    #       the same grammatical structure for two or more related elements in a
    #       sentence.
    #       - If you have nothing to add, do not add anything!"""

    # randomize the order
    shuffled_agents = agents
    random.shuffle(shuffled_agents)

    # write!
    story: Story = Story()
    for i in range(0, num_iterations):
        # invoke each agent for a turn at the pen
        for agent in shuffled_agents:
            summary_prompt = dedent(
                text=f"""
                This is my understanding of what this story is about.
                I will use this material to guide how I write the story.

                ###
                {agent["consensus_summary"]}
                ###
                """
            )

            story_prompt = dedent(
                text=f"""
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
                            AssistantMessage(summary_prompt),
                            UserMessage(story_prompt),
                        ],
                        max_retries=5,
                        abort_signal=AbortSignal.timeout(duration=TIMEOUT_DURATION),
                    )
                    success = True
                except AbortError:
                    attempt = attempt + 1
                    log(
                        f"[red bold]→ Writer {agent['name']} timed out, trying again (attempt {attempt})[/]"
                    )

            if result is None:
                raise AbortError(
                    "Writing phase failed to complete within the allotted time."
                )

            new_sentence = result.object.get("sentence", "")

            if new_sentence is None or new_sentence == "":
                log(f"[green bold]Writer {agent['name']} has yielded[/]")
            else:
                # check this against all the previous sentences
                # if we find an exact match, the author loses their turn
                is_new_sentence = True
                for s in story.story:
                    if s.sentence.lower() == result.object.get("sentence", "").lower():
                        log(
                            f"[red bold]→ Writer {agent['name']} generated a pre-existing sentence, ignoring[/]"
                        )
                        is_new_sentence = False
                        break

                if is_new_sentence:
                    log(
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

    success = False
    attempt = 0
    result = None
    while not success:
        try:
            result = await llm.create_structure(
                schema=Title,
                messages=[
                    SystemMessage(title_prompt),
                    UserMessage(f"This is the story:\n###\n{story.render()}\n###"),
                ],
                max_retries=5,
                abort_signal=AbortSignal.timeout(duration=TIMEOUT_DURATION),
            )
            success = True
        except AbortError:
            attempt = attempt + 1
            log(
                f"[red bold]→ Title generation timed out, trying again (attempt {attempt})[/]"
            )

    if result is None:
        raise AbortError(
            "Title generation failed to complete within the allotted time."
        )

    title = result.object.get("title", "")
    return title


async def main() -> None:
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
    parser.add_argument(
        "--style",
        type=str,
        required=False,
        help="Path to file containing style instructions for the Writing phase",
    )
    parser.add_argument(
        "--log-dir", type=str, required=False, help="Path to log directory"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the LLM (default: 0.2)",
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging(log_dir=args.log_dir)

    # Read story prompt from file
    with open(args.story_prompt, "r") as file:
        story_prompt = file.read().strip()

    # Read style instructions
    if args.style is not None:
        with open(args.style, "r") as file:
            style_instructions = file.read().strip()

    # Read prompt files and create agents
    agents = []
    for file_path in args.prompt_files:
        with open(file_path, "r") as file:
            name = file_path.split("/")[-1].split(".")[0]
            position_statement = file.read().strip()
            (model_name, position_statement) = parse_model_name_from_position_statement(
                position_statement=position_statement
            )
            llm = create_llm(model_name=model_name, temperature=args.temperature)
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
        other_agent_names = all_agent_names.copy()
        other_agent_names.discard(agent["name"])
        prompt = dedent(
            text=f"""You are {agent["name"]} and you are in a
                  writer's room with {", ".join(other_agent_names)}.
                  Your task is to write a story together on the following
                  topic.

                  ###
                  {story_prompt}
                  ###

                  Your position statement on this topic is below.

                  ###
                  {agent["position_statement"]}
                  ###
                  """
        )
        agent["system_prompt"] = prompt

    # Print agent information
    log("[bold cyan]Welcome to Story Arena![/]")
    log(f"[grey35]Temperature: {args.temperature}[/]")
    log(f"[grey35]Story prompt: {story_prompt}[/]")
    for agent in agents:
        log(
            f"[cyan]Writer {agent['name']} has entered the arena![/] [grey35]→ {agent['model_name']}[/]"
        )

    # Run the workflow
    log(f"[bold cyan]{'*' * 72}[/]")
    log("[bold cyan]Starting Ideation Phase...[/]")
    for agent in agents:
        log(f"[cyan][bold]Ideation[/bold]: {agent['name']}'s turn[/]")
        response = await run_ideation_phase(agent=agent)
        log(f"[magenta][bold]{agent['name']}[/bold]: {response}[/]")
        agent["ideation_result"] = response

    log(f"[bold cyan]{'*' * 72}[/]")
    log("[bold cyan]Starting Consensus Phase...[/]")
    _ = await run_consensus_phase(agents=agents)

    log(f"[bold cyan]{'*' * 72}[/]")
    log("[bold cyan]Starting Writing Phase...[/]")
    story = await run_writing_phase(agents=agents, style_instructions=style_instructions, num_iterations=7)

    # use the first agent's llm to pick the title
    llm = agents[0]['llm']
    title = await generate_story_title(llm=llm, story=story)

    log(f"[bold cyan]{'*' * 72}[/]")
    log(f"[bold cyan]{title}[/]")
    log(f"[cyan]by {', '.join(agent['name'] for agent in agents)}[/]")
    log("")
    log(f"[cyan]{story.render()}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
