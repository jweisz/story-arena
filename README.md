# ✏️ Story Arena

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![mise](https://img.shields.io/badge/tasks-mise-6E56CF)](https://mise.jdx.dev/)
[![LiteLLM](https://img.shields.io/badge/LLMs-LiteLLM-1F6FEB)](https://docs.litellm.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Story Arena puts several LLM-powered writers in a virtual writers' room and has them write a short story together. Each writer has its own persona and point of view (a *position statement*), and each one can run on a different model and backend: a local Ollama model can argue plot points with Claude and GPT.

Writers go through three phases. First they **ideate** on their own. Then they argue their way to a **consensus** on the story's direction. Finally they **write** the story one sentence at a time.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Configuring models](#configuring-models)
- [Writers, prompts, and styles](#writers-prompts-and-styles)
- [Command-line reference](#command-line-reference)
- [Tasks](#tasks)
- [Output and logs](#output-and-logs)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## How it works

1. **Ideation.** Each writer reads the story prompt and their own position statement, then brainstorms a genre, setting, characters, story shape (in the style of [Vonnegut's story shapes](https://bigthink.com/high-culture/vonnegut-shapes/)), plot, and conflict.
2. **Consensus.** The writers take turns in random order, discussing and defending their ideas. Each writer must take at least 2 turns before they can say "I yield". The phase ends when every writer has yielded, or after 10 turns per writer. Each writer then summarizes their own reading of what was decided. Writers may disagree with the group.
3. **Writing.** The writers take turns adding one sentence at a time to the story, guided by their summary and an optional style guide. Each writer gets 7 turns. If a writer repeats a sentence that's already in the story, that sentence is thrown out. Finally, the first writer's model comes up with a title.

Every LLM call except the summaries uses structured output (JSON matching a Pydantic schema). Calls time out after 120s and are retried up to 5 times.

## Requirements

- [mise](https://mise.jdx.dev/getting-started.html) (recommended). It installs the right Python and uv for you. Without mise you need Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Access to at least one model backend:
  - [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) for local models (no API key needed), and/or
  - an [Anthropic](https://console.anthropic.com/) or [OpenAI](https://platform.openai.com/) API key.

> [!NOTE]
> Python 3.14+ isn't supported yet, because one of LiteLLM's native dependencies doesn't build on it. The repo pins Python 3.13 via `.python-version`.

## Quickstart

```bash
git clone https://github.com/jweisz/story-arena.git
cd story-arena

mise install          # install Python 3.13 + uv
mise run install      # create .venv and install dependencies

cp .env.sample .env   # then add API keys for the providers you'll use
```

Run the software-engineering scenario with every writer on Claude:

```bash
mise run swe -- --model claude
```

Or run it entirely on your machine with Ollama:

```bash
ollama pull granite4.2:3b
mise run swe -- --model-override local
```

<details>
<summary>Without mise</summary>

```bash
uv sync
uv run story_arena.py writers/JW.txt writers/KV2.txt writers/MM2.txt writers/AI.txt \
  --story-prompt prompts/future_swe_complex_v4.txt \
  --style styles/martha_wells.txt \
  --model claude
```

</details>

## Configuring models

Story Arena calls models through [LiteLLM](https://docs.litellm.ai/), so it works with **any LiteLLM chat model**. A model is identified by a LiteLLM model string (`<provider>/<model>`) or by an alias defined in [`models.toml`](models.toml).

### Supported backends

| Backend | Model string | Credentials / endpoint | Example |
| --- | --- | --- | --- |
| [Ollama](https://ollama.com) | `ollama_chat/<model>` | `OLLAMA_API_BASE` (default `http://localhost:11434`) | `ollama_chat/granite4.2:3b` |
| [LM Studio](https://lmstudio.ai) | `lm_studio/<model id>` | `LM_STUDIO_API_BASE` (e.g. `http://localhost:1234/v1`), or `api_base` in `models.toml` | `lm_studio/qwen/qwen3-8b` |
| [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models) | `anthropic/<model>` | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-5-5` |
| [OpenAI](https://platform.openai.com/docs/models) | `openai/<model>` | `OPENAI_API_KEY` | `openai/gpt-5.6` |

Other LiteLLM providers (Gemini, Mistral, Bedrock, watsonx, OpenRouter, any OpenAI-compatible server, …) work the same way. See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for the prefixes and environment variables. API keys are read from your environment or from `.env`.

> [!IMPORTANT]
> Writers must return structured JSON, so pick models that support structured output / JSON mode. Very small local models sometimes return malformed JSON. Story Arena retries those calls, but a stronger model gives better results.

### Choosing a model for each writer

A writer's model comes from the first source that sets one, in this order:

1. `--model-override <model>` forces every writer onto one model.
2. A header on the first line of the writer file, e.g. `# claude` or `# ollama_chat/granite4.2:3b`.
3. `--model <model>` is the default for writers whose file has no header.

If none of these sets a model, Story Arena stops and tells you which writer file needs one. Writer files that still use the old `# ollama:<model>` header are mapped to `ollama_chat/<model>` automatically.

### Model aliases (`models.toml`)

`models.toml` defines short names for models and holds per-model settings. Run `mise run models` to list them. The included aliases are:

| Alias | Model |
| --- | --- |
| `local` | `ollama_chat/granite4.2:3b` (thinking disabled for speed; uses `OLLAMA_API_BASE`) |
| `lmstudio` | `lm_studio/qwen/qwen3-8b`: change this to the model you have loaded |
| `claude`, `claude-opus`, `claude-haiku` | `anthropic/claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5` |
| `gpt`, `gpt-mini` | `openai/gpt-5.6`, `openai/gpt-5.4-mini` |

To use Ollama on another machine, point `OLLAMA_API_BASE` at it, either in `.env` or for a single run:

```bash
OLLAMA_API_BASE=http://my-server.local:11434 mise run swe -- --model local
```

Add your own:

```toml
[models.my-server]
model = "openai/my-finetune"            # required: LiteLLM model string
api_base = "https://llm.example.com/v1" # optional: custom endpoint
api_key_env = "MY_SERVER_API_KEY"       # optional: env var holding the key
temperature = 0.7                       # optional: --temperature overrides it
max_tokens = 1024                       # any other key is passed to litellm.acompletion()
```

Use a different file with `--models-config path/to/models.toml`.

## Writers, prompts, and styles

| Directory | Contents |
| --- | --- |
| `writers/` | One file per writer. The file name (without `.txt`) becomes the writer's name. The text is that writer's persona or position statement. An optional first line `# <model>` picks the writer's model. |
| `prompts/` | Story prompts: the topic the room is writing about. |
| `styles/` | Optional style guides, applied during the Writing phase (e.g. "write like Martha Wells"). |

Example writer file:

```text
# claude
You are a traditionalist who believes that software engineering is best left to humans...
```

The `swe` task uses the scenario from our paper, where the writers imagine a future in which AI has changed how software engineers work. It uses the writers `JW`, `KV2`, `MM2` and `AI`, the prompt `prompts/future_swe_complex_v4.txt` and the style `styles/martha_wells.txt`. The other writers (`AntagonisticAnnie`, `FickleFannie`, `LogicalLisa`, `UncompromisingUlysses`, …) are personality archetypes you can mix in.

## Command-line reference

```text
uv run story_arena.py [options] WRITER_FILE [WRITER_FILE ...]
```

| Option | Description |
| --- | --- |
| `--story-prompt PATH` | **Required.** File containing the story prompt. |
| `--model MODEL` | Default model for writers without a `# model` header. |
| `--model-override MODEL` | Use this model for every writer. |
| `--models-config PATH` | Model alias file (default: `models.toml`). |
| `--list-models` | Print the configured aliases and exit. |
| `--style PATH` | File containing style instructions for the Writing phase. |
| `--temperature FLOAT` | Sampling temperature for every writer, overriding any `temperature` in `models.toml`. If neither is set, each model uses its own default (recommended: model publishers tune these, e.g. Ollama's `qwen3:8b` defaults to 0.6). Models that don't accept a temperature, such as some reasoning models, ignore it. |
| `--log-dir PATH` | Where to write logs (default `runs/run-<timestamp>/`). |

## Tasks

| Task | What it does |
| --- | --- |
| `mise run install` | `uv sync`: create `.venv` and install dependencies |
| `mise run story -- <args>` | Run Story Arena with any arguments |
| `mise run swe -- --model <m>` | Run the software-engineering scenario |
| `mise run models` | List model aliases from `models.toml` |
| `mise run lint` | Ruff lint + format check |
| `mise run format` | Auto-fix lint issues and format |

## Output and logs

Progress is printed to the terminal as it happens. Each run also writes a log directory, `runs/run-<timestamp>/` by default (ignored by git), containing:

- `console.log`: the full transcript (ideas, conversation, story)
- `prompts.log`: every prompt sent to the models
- `debug.log`: HTTP and LiteLLM debug output

## Project structure

```text
story_arena.py   # CLI and the three-phase workflow
llm.py           # LiteLLM wrapper: model aliases, structured output, retries
models.toml      # model aliases
mise.toml        # tool versions and tasks
pyproject.toml   # project metadata and dependencies (managed with uv)
writers/         # writer personas
prompts/         # story prompts
styles/          # writing-style guides
```

## Contributing

Issues and pull requests are welcome at <https://github.com/jweisz/story-arena>.

1. Fork the repo and create a branch.
2. `mise run install`
3. Make your change, then run `mise run format` and `mise run lint`.
4. Open a pull request describing what you changed and how you tested it (for example, a short `mise run swe -- --model-override local` run).

## License

Story Arena is licensed under the [Apache License 2.0](LICENSE).
