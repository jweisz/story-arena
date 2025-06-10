# Story Arena

Story Arena is a collaborative story-writing tool that leverages multiple AI models to generate stories through a three-phase process: ideation, consensus, and writing.

## Installation

To use Story Arena, you will need to have Python installed on your system. The project uses Poetry for dependency management. Follow these steps to set up the project:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run `poetry install` to install the required dependencies.

## Usage

Story Arena is a command-line tool. To use it, you will need to provide the paths to agent prompt files and a story prompt file as arguments.

```bash
poetry run python3 story_arena.py --story-prompt <story_prompt_file> <agent_prompt_files>
```

- `<story_prompt_file>`: Path to the file containing the story prompt.
- `<agent_prompt_files>`: Paths to one or more agent prompt files.

### Configuration

The tool uses environment variables for configuration, such as API keys for different AI models. You can set these variables in a `.env` file in the project root.

## Project Structure

The project is structured into several key components:

- `story_arena.py`: The main script that runs the story-writing process.
- `prompts/`: Directory containing files for story prompts.
- `writers/`: Directory containing files for writer prompts.

## How it Works

Story Arena uses a three-phase process to generate stories collaboratively with multiple AI models.

1. **Ideation Phase**: In this initial phase, each AI model generates ideas for the story based on its specific prompt and the overall story prompt. The models produce detailed concepts including genre, setting, characters, story shape, plot, and conflict.

2. **Consensus Phase**: After ideation, the models engage in a consensus-building discussion. They share their ideas, debate, and agree on the key elements of the story. This phase ensures that the story has a coherent direction and that all models are aligned.

3. **Writing Phase**: With a consensus reached on the story elements, the models collaboratively write the story. They take turns adding sentences to the narrative, ensuring a cohesive and engaging story. The writing process is guided by the agreed-upon story elements and follows specific writing rules to maintain quality and consistency.

## Logging

The tool logs its progress and output to a directory specified by the `--log-dir` argument. If not provided, it creates a log directory based on the current timestamp.

## Supported Models

The following models are currently supported:

- rits:deepseek-v2-5
- rits:granite-3-3-8b-instruct
- rits:granite-4-0-small-franconia
- rits:llama-4-mvk-17b-128e-fp8
- rits:llama-4-scout-17b-16e
- rits:microsoft-phi-4
- rits:mistral-small-3-1-24b-2503
- rits:qwen2-5-72b-instruct
- watsonx:granite-3-3-8b-instruct
- watsonx:llama-4-maverick-17b-128e-instruct-fp8
- watsonx:mistral-large
- watsonx:mistral-small-3-1-24b-instruct-2503

## Contributing

Contributions to Story Arena are welcome. Please submit pull requests or issues through the GitHub repository.
