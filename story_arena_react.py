import argparse
import asyncio
import logging
import os
from enum import StrEnum, auto
from typing import Any, Self

from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import AnyTool
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolOutput, ToolRunOptions
from beeai_framework.utils.cancellation import AbortSignal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console

# Rich console
console = Console()

# Enable logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)

# Tool definition for Writing phase
class Sentence(BaseModel):
    sentence: str = Field(description="A sentence in a story.")
    author: str = Field(description="The author (or authors) of the sentence.")

    def __str__(self) -> str:
        return f"{self.sentence} [{self.author}]"

class EditOperationType(StrEnum):
    Add = auto()
    Replace = auto()
    Delete = auto()

class StoryToolInput(BaseModel):
    author: str = Field(description="The author who is making this edit to the story.")
    operation: EditOperationType = Field(description="The edit operation to perform: Add, Replace, Delete")
    sentence: Sentence = Field(description="For Add operations, the new sentence to add. For Replace operations, the replacement sentence. For Delete operations, the sentence to delete.")
    index: int = Field(description="For Add operations, the index position in the story where the sentence should be added. For Replace operations, the index of the sentence to be replaced. For Delete operations, the index of the sentence to be deleted.")

class StoryToolOutput(ToolOutput):
    def __init__(self, change_summary: str) -> None:
        super().__init__()
        self.change_summary = change_summary

    def get_text_content(self) -> str:
        return self.change_summary

    def is_empty(self) -> bool:
        return not self.change_summary

class EditStoryTool(Tool[StoryToolInput, ToolRunOptions, StoryToolOutput]):
    name = "EditStoryTool" # type: ignore
    description = "Make an edit to the story, either by adding, replacing, or deleting a sentence." # type: ignore
    input_schema = StoryToolInput # type: ignore

    def __init__(self, story: list[Sentence]) -> None:
        super().__init__()
        self.story = story

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "story_arena", "editstorytool"],
            creator=self,
        )

    async def _run(
        self, input: StoryToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> StoryToolOutput:
        change_summary = ""
        if input.operation == EditOperationType.Add:
            # add sentence at index
            self.story.insert(input.index, input.sentence)
            change_summary = f"Added sentence at index {input.index}: {input.sentence}. [{input.author}]"
        elif input.operation == EditOperationType.Replace:
            # replace sentence at index
            self.story[input.index] = input.sentence
            # maintain authorship
            self.story[input.index].author = ", ".join([input.sentence.author, self.story[input.index].author])
            change_summary = f"Changed sentence at index {input.index}: {input.sentence}. [{self.story[input.index].author}]"
        elif input.operation == EditOperationType.Delete:
            # delete sentence at index
            self.story.pop(input.index)
            change_summary = f"Deleted sentence at index {input.index}"

        return StoryToolOutput(change_summary=change_summary)

    async def clone(self) -> Self:
        cloned = await super().clone()
        return cloned

# Ideation phase for a single agent
async def run_ideation_phase(agent) -> str:
    ideation_prompt = """
        You are in the Ideation phase of the story-writing process.
        During this phase, your task is to brainstorm story ideas including setting, characters, and conflict.
        Make sure that your ideas conform to your position statement.
        """

    class IdeationResult(BaseModel):
        setting: str = Field(description="The setting of the story, including the location, season, era, etc.")
        characters: str = Field(description="A list of characters who are involved in the story with relevant backstories and personal details")
        conflict: str = Field(description="A description of a type of conflict between one or more of the characters")
        other_elements: str = Field(description="Other elements that are pertinent to the kind of story you want to tell")

    result = await agent['llm'].create_structure(
        schema=IdeationResult,
        messages=[
            SystemMessage(agent['system_prompt']),
            UserMessage(ideation_prompt),
        ],
        max_retries=3,
        abort_signal=AbortSignal.timeout(duration=30)
    )

    ideation_result = f"""This is your idea for the story:
        Setting: {result.object['setting']}
        Characters: {result.object['characters']}
        Conflict: {result.object['conflict']}
        Other elements: {result.object['other_elements']}
        """

    return ideation_result

# Consensus phase with round-robin conversation
async def run_consensus_phase(agents, num_agent_turns=20) -> list[UserMessage]:
    max_turns = num_agent_turns * len(agents)
    consensus_prompt = f"""
        You are in the consensus phase of the story-writing process. The way this phase
        works is that you will have a conversation with the other writers in the
        writing room about the ideas you brainstormed in the ideation phase. The goal
        is to argue for your ideas on the details of the story (setting, characters, conflict, etc.),
        while remaining open to the arguments of the other writers. It is important to argue for choices
        that align and support your position statement. It is also important to come to consensus with
        your fellow writers. How you achieve this balance in the face of conflicts is up to you.
        If you are satisfied with all of the decisions that have been made, tell the group "I yield" to
        indicate that you are happy with the decisions made by the group. If the group does not reach
        consensus after {max_turns} turns (you will get {num_agent_turns} turns), the
        consensus phase will end, possibly with some decisions left unmade. Do your best
        to reach consensus before this occurs.
        """

    active_agents = set(agent['name'] for agent in agents)
    turn = 0

    # create the conversational space
    conversation: list[UserMessage] = []

    while len(active_agents) > 1:
        # give each agent a turn
        for agent in agents:
            # make sure they are still active
            if agent['name'] in active_agents:
                turn = turn + 1

                console.print(f"[cyan][bold][{turn}] {agent['name']}'s turn[/bold][/]")

                result = await agent['llm'].create(
                    messages=[
                        SystemMessage(agent['system_prompt']),
                        UserMessage(consensus_prompt),
                        *conversation
                    ],
                    max_retries=3,
                    abort_signal=AbortSignal.timeout(duration=30)
                )

                # pull out the last message
                message = result.messages[-1].content[-1].text
                conversation.append(UserMessage(f"{agent['name']}: {message}"))
                console.print(f"[magenta][bold]{agent['name']}[/bold]: {message}[/]")

                if "i yield" in message.lower():
                    console.print(f"[green bold]Writer {agent['name']} has yielded[/]")
                    active_agents.remove(agent['name'])

                if turn > max_turns:
                    console.print(f"[green bold]Turn number ({turn}) exceeds turn limit ({max_turns}), ending conversation[/]")
                    active_agents.clear()

    console.print("[green bold]All writers have yielded, moving to summarization[/]")

    # each agent summarizes the decisions made in the conversation
    for agent in agents:
        summary_prompt = """
            This is the conversation you just had with the other writers in the writing room. Produce a summary
            of the major decisions that were made. You will use this summary to aid your writing process.
            """
        console.print(f"[cyan][bold]{agent['name']}'s summary[/bold][/]")
        result = await agent['llm'].create(
            messages=[
                SystemMessage(agent['system_prompt']),
                UserMessage(summary_prompt),
                *conversation
            ],
            max_retries=3,
            abort_signal=AbortSignal.timeout(duration=30)
        )

        summary = result.messages[-1].content[-1].text
        console.print(f"[magenta][bold]{agent['name']}[/bold]: {summary}[/]")
        agent['consensus_summary'] = summary
    return conversation

# Writing phase with round-robin conversation
async def run_writing_phase(agents, num_iterations=10) -> list[Sentence]:
    # def process_agent_events(data: Any, event: EventMeta) -> None:
    #     """Process agent events and log appropriately"""

    #     if event.name == "error":
    #         logging.debug("Agent 🤖 : ", FrameworkError.ensure(data.error).explain())
    #     elif event.name == "retry":
    #         logging.debug("Agent 🤖 : ", "retrying the action...")
    #     elif event.name == "update":
    #         logging.debug(f"🤖: {data.data.thought}")
    #     elif event.name == "start":
    #         logging.debug("Agent 🤖 : ", "starting new iteration")
    #     elif event.name == "success":
    #         logging.debug("Agent 🤖 : ", "success")

    writing_prompt = f"""
        You are in the writing phase of the story-writing process. Your job is to work with the other
        writers in the room to construct a story based on your own understanding of the consensus
        reached during the consensus phase. Remember that a story is just a collection of sentences.
        You may add a sentence anywhere within the story. You may rewrite an existing sentence.
        You may delete a sentence. You will work in round-robin fashion with your fellow writers.
        You will have {num_iterations} iterations of writing. Use them well. After each iteration,
        tell your fellow co-authors what you did.
    """

    # instantiate ReAct agents
    story: list[Sentence] = []
    for agent in agents:
        tools: list[AnyTool] = [
            EditStoryTool(story=story)
        ]
        agent['writing_agent'] = ReActAgent(llm=agent['llm'], tools=tools, memory=UnconstrainedMemory())

    # write!
    last_agent = None
    last_edit_comment = None
    for i in range(0, num_iterations):
        # invoke each agent for a turn at the pen
        for agent in agents:
            story_txt = "\n".join([f"{i}. {s}" for (i, s) in enumerate(story)])
            prompt = f"""
                {agent['system_prompt']}
                {writing_prompt}

                It is your turn to edit the story. You are on iteration {i} of {num_iterations}.
                This is the story so far. Each sentence is prefixed with a number and includes the
                author (or authors) of that sentence in square brackets at the end.
                Use the line numbers when deciding what edits to make.
                {story_txt}
                """

            if last_agent is not None and last_edit_comment is not None:
                prompt = (
                    prompt +
                    f"""\nThe last edit was made by {last_agent},
                    who said this about what they did:
                    {last_edit_comment}\n
                    """
                )

            react = agent['writing_agent']
            response = await react.run(
                prompt=prompt,
                execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=5, max_iterations=10),
            )
            # .on("*", process_agent_events, EmitterOptions(match_nested=False))

            last_agent = agent['name']
            last_edit_comment = response.result.text
    return story

async def main() -> None:
    load_dotenv()
    RITS_API_KEY = os.getenv('RITS_API_KEY')
    RITS_API_URL = os.getenv('RITS_API_URL', 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com')
    MODEL_NAME = os.getenv('MODEL_NAME')

    if RITS_API_KEY is None or RITS_API_URL is None or MODEL_NAME is None:
        raise ValueError("RITS_API_KEY, RITS_API_URL, and MODEL_NAME must be defined in the .env file.")

    parser = argparse.ArgumentParser(description='Multi-Agent Story Writer')
    parser.add_argument('prompt_files', type=str, nargs='+', help='Paths to agent prompt files')
    parser.add_argument('--story-prompt', type=str, required=True, help='Path to file containing the story prompt')
    args = parser.parse_args()

    # Initialize LLM
    if MODEL_NAME.startswith("ollama:"):
        llm = ChatModel.from_name(MODEL_NAME)
    elif MODEL_NAME.startswith("openai:"):
        settings = {
            "extra_headers": {"RITS_API_KEY": RITS_API_KEY},
            "api_key": RITS_API_KEY,
            "base_url": RITS_API_URL
        }
        llm = OpenAIChatModel(model_id=MODEL_NAME.split(":")[1], settings=settings)
    else:
        raise ValueError("Invalid MODEL_NAME. It should start with 'ollama:' or 'openai:'")

    # Read story prompt from file
    with open(args.story_prompt, 'r') as file:
        story_prompt = file.read().strip()

    # Read prompt files and create agents
    agents = []
    for file_path in args.prompt_files:
        with open(file_path, 'r') as file:
            name = file_path.split('/')[-1].split('.')[0]
            position_statement = file.read().strip()
            agents.append({'name': name,
                           'position_statement': position_statement,
                           'llm': llm})

    all_agent_names = set([a['name'] for a in agents])
    for agent in agents:
        other_agent_names = all_agent_names - set(agent['name'])
        prompt = f"""
        You are {agent['name']} and you are in a writer's room with {', '.join(other_agent_names)}.
        Your task is to write a story together on this topic:
        {story_prompt}
        Your position statement is:
        {agent['position_statement']}
        """
        agent['system_prompt'] = prompt

    # Print agent information
    console.print("[bold cyan]Welcome to Story Arena![/]")
    for agent in agents:
        console.print(f"[cyan]Writer {agent['name']} has entered the arena![/]")

    # Run the workflow
    console.print(f"[bold cyan]{'*'*72}[/]")
    console.print("[bold cyan]Starting Ideation Phase...[/]")
    for agent in agents:
        console.print(f"[cyan][bold]Ideation[/bold]: {agent['name']}'s turn[/]")
        response = await run_ideation_phase(agent=agent)
        console.print(f"[magenta][bold]{agent['name']}[/bold]: {response}[/]")
        agent['ideation_result'] = response

    console.print(f"[bold cyan]{'*'*72}[/]")
    console.print("[bold cyan]Starting Consensus Phase...[/]")
    _ = await run_consensus_phase(agents=agents)

    console.print(f"[bold cyan]{'*'*72}[/]")
    console.print("[bold cyan]Starting Writing Phase...[/]")
    story = await run_writing_phase(agents=agents, num_iterations=10)

    console.print(f"[bold cyan]{'*'*72}[/]")
    console.print("[bold cyan]Final Story[/]")
    for sentence in story:
        console.print(f"[magenta]{sentence.sentence} [bold][{sentence.author}][/bold][/magenta]")

if __name__ == "__main__":
    asyncio.run(main())
