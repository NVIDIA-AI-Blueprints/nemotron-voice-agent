# Configure Prompts

You can customize your voice agent's personality, behavior, and response format using system prompts. Built-in prompts are defined in example-local `prompts.yaml` files and can be switched or added at runtime via the UI.

## Switching / Adding Prompts via the UI

The client UI includes a prompt selector dropdown. Select any prompt to switch during a session. The change takes effect on the next conversation turn.

You can add custom prompts directly from the UI's **Prompts** tab without editing any files or restarting the server. Custom prompts added in the UI are stored in the browser's localStorage, so they persist for that browser/profile and origin only.

> **Note:** Not every prompt is editable in every example. In multi-agent examples like the Omni Subagents example, only the top-level session prompt is switchable from the UI. The per-agent prompts (defined under `agent_prompts:` in the example's `prompts.yaml`) are bound to their agent and are not selectable or editable in the UI. To change those, edit the example's `prompts.yaml` and redeploy.

## Available Prompt Presets

Prompt presets are defined per example. The Generic Cascaded example currently provides the presets below (the active default is set by `defaults.prompt` in `examples_registry.yaml`).

| Prompt Key | Description |
|------------|-------------|
| `generic_assistant` | Generic voice assistant with tool support and a single-sentence response format. |
| `generic_assistant_without_tools` | Generic voice assistant without tool access and with a single-sentence response format. |
| `multi_speaker_assistant` | Listens to up to eight speakers and answers only when called by name (Nemotron or assistant). Requires speaker labels. Refer to [Enable Speaker Labels](enable-speaker-labels.md). |
| `flowershop` | Flora persona for the GreenForce Garden flower-shop scenario with strict flow rules. |

## Changing the Default Prompt

Set the per-example default with `defaults.prompt` in `examples_registry.yaml`. As a fallback (when that key is not one of the active example's prompt keys in `prompts.yaml`), the entry marked `default: true` is used, otherwise the first entry.

```yaml
my_prompt:
  default: true
  description: "Your prompt description"
  content: |
    ...
```

## Adding Built-In Prompts via `prompts.yaml`

To make a prompt available as a built-in option for all users of an example, add an entry to that example's `prompts.yaml`, such as [`src/examples/generic/prompts.yaml`](../../src/examples/generic/prompts.yaml). The client loads built-in prompts for the active example. Refresh open browser tabs after editing YAML.

```yaml
my_custom_prompt:
  description: "Your prompt description"
  content: |
    Your system prompt here...
    Define personality, rules, and response format.
```

## Frontend/Backend Live Prompt Catalog

A `PROMPT_FILE_PATH` override replaces the full catalog. Start from the shipped [Live prompts.yaml](../../src/examples/frontend_backend_live/prompts.yaml) so the `Cascade`, `Lines`, and `Core` entries stay aligned with the tool and event schemas.

Use `session.instructions` for identity, speaking style, and when to delegate. Describe the backend's capabilities briefly. Put detailed procedures in `delegation.responses.instructions` and function schemas in `delegation.responses.tools`. For example, tell the frontend that it can delegate reservation changes. Give the backend the lookup and confirmation rules. Use `session.update` to replace the tool set for new work.

Keep the current Core schemas for `caller_message`, `resolve_questions`, `set_work_state`, and `submit_decision`. The Writer's `Cascade.commentary` and `Cascade.answer` entries should preserve the result, question, and actor for each step. Use short sentences and place the requested result first.

Custom catalogs for [reconnectable calls](use-live-gateway.md#resume-a-dropped-websocket-call) need a nonblank `Cascade.attachment_note` template with a plain `{text}` placeholder. Copy the shipped wording to preserve uncertainty about interrupted speech.

## Best Practices for Voice Prompts

Use the following practices when writing voice prompts:

- Keep responses concise (1-2 sentences).
- Avoid special characters like `*`, `-`, `/` in output.
- Avoid bullet points or numbered lists (breaks voice flow).
- Define clear output format for structured data.
- Use plain text only.
