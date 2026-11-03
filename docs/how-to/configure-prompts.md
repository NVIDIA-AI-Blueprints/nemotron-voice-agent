# Configure Prompts

You can customize your voice agent's personality, behavior, and response format using system prompts. Built-in prompts are defined in example-local `prompts.yaml` files and can be switched or added at runtime via the UI. The Frontend/Backend Live example is the exception: its built-in prompts live in prompt profile folders under `src/examples/frontend_backend_live/prompts/`, which the `prompt_catalog` key of its `examples_registry.yaml` entry serves to the UI. Refer to [Prompt Profiles](../../src/examples/frontend_backend_live/README.md#prompt-profiles).

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

Set the per-example default with `defaults.prompt` in `examples_registry.yaml`. As a fallback (when that key is not one of the active example's prompt keys in `prompts.yaml`), the entry marked `default: true` is used, otherwise the first entry. For the Frontend/Backend Live example, `defaults.prompt` names a prompt profile, `cafe`.

```yaml
my_prompt:
  default: true
  description: "Your prompt description"
  content: |
    ...
```

## Adding Built-In Prompts via `prompts.yaml`

To make a prompt available as a built-in option for all users of an example, add an entry to that example's `prompts.yaml`, such as [`src/examples/generic/prompts.yaml`](../../src/examples/generic/prompts.yaml). The client loads built-in prompts for the active example. Refresh open browser tabs after editing YAML. To add a built-in prompt to the Frontend/Backend Live example, add a prompt profile folder instead, as [Prompt Profiles](../../src/examples/frontend_backend_live/README.md#prompt-profiles) describes.

```yaml
my_custom_prompt:
  description: "Your prompt description"
  content: |
    Your system prompt here...
    Define personality, rules, and response format.
```

## Best Practices for Voice Prompts

- Keep responses concise (1-2 sentences).
- Avoid special characters like `*`, `-`, `/` in output.
- Avoid bullet points or numbered lists (breaks voice flow).
- Define clear output format for structured data.
- Use plain text only.
