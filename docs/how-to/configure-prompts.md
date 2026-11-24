# Configure Prompts

You can customize your voice agent's personality, behavior, and response format using system prompts. Built-in prompts are defined in example-local `prompts.yaml` files and can be switched or added at runtime through the UI. This guide applies to the Pipecat examples (`nva-pipecat`).

## Switching / Adding Prompts via the UI

Use the **Prompt** dropdown before connecting to choose the preset for your next browser session.

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

To make a prompt available as a built-in option for all users of an example, add an entry to that example's `prompts.yaml`, such as [`src/nemotron_voice_agent/pipecat/examples/generic/prompts.yaml`](../../src/nemotron_voice_agent/pipecat/examples/generic/prompts.yaml). The client loads built-in prompts for the active example. Refresh open browser tabs after editing YAML.

```yaml
my_custom_prompt:
  description: "Your prompt description"
  content: |
    Your system prompt here...
    Define personality, rules, and response format.
```

## Frontend/Backend Live Prompt Catalog

The **Prompts** tab shows the configured prompt sections in request order for the Talker, Writer, backend worker, and conductor when enabled. It labels where the request includes runtime context such as conversation history, work status, and tool results.

Before connecting, open an editable section and select **Edit**, then **Save**. Saved edits apply to new sessions for this example in the same browser. If a section appears in several calls, the edit applies to each call. Select **Reset** to restore the section’s built-in text. Edit the YAML to change the server defaults.

A `PROMPT_FILE_PATH` override replaces the full catalog. When upgrading, copy the current [Live prompt catalog](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/prompts.yaml) and reapply your custom instructions. Keep its required `Cascade`, `Lines`, and `Core` entries and tool schemas. Restart the server after editing Core entries or the default backend prompt.

Use `session.instructions` for speaking style and a short description of what to delegate, such as checking availability or changing a booking. In Responses mode, put business rules in `delegation.responses.instructions` and function schemas in `delegation.responses.tools`. Apply disclosure rules to both sides because the frontend can speak a current backend result as written if rewording fails. For session updates, refer to [Live Session Settings](use-live-gateway.md#session-settings).

For the experimental `tools` talker style, keep `Cascade.tool_frame_before`, `tool_routing`, `tool_rules`, and the `tools.delegate` and `tools.wait` schemas. Reconnectable calls also need the `Cascade.attachment_note` template with its plain `{text}` placeholder. Refer to [Talker Style](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md#talker-style) and [Resume a Dropped WebSocket Call](use-live-gateway.md#resume-a-dropped-websocket-call).

## Best Practices for Voice Prompts

Use the following practices when writing voice prompts:

- Keep responses concise (1-2 sentences).
- Avoid special characters like `*`, `-`, `/` in output.
- Avoid bullet points or numbered lists (breaks voice flow).
- Define clear output format for structured data.
- Use plain text only.
