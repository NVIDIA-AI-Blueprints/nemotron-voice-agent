# Nemotron Voice Agent Documentation Guidance

## Role

Maintain accurate, task-focused documentation for developers who evaluate,
configure, deploy, extend, and troubleshoot Nemotron Voice Agent. Verify every
command, profile, service name, model identifier, default, port, hardware
requirement, and configuration key against checked-in repository sources.

## Writing Style Guide

Apply these rules to documentation, examples, headings, UI text, and release
notes that you create or edit.

- Write in a professional, active, conversational, and engaging voice.
- Use active voice whenever possible. Use present tense for product behavior.
  Address the reader in second person as "you."
- Keep sentences concise. Prefer sentences with fewer than 30 words.
- Use plain English and precise technical terms. Avoid jargon, filler,
  colloquialisms, and flowery marketing claims.
- Avoid contractions in technical documentation. Write "do not," "cannot,"
  and "it is."
- Write "NVIDIA" in all caps and use "an NVIDIA," not "a NVIDIA."
- Spell out uncommon abbreviations on first use. Spell out LLM, RAG, SLM, VLM,
  and MoE on first use.
- Use NVIDIA spellings such as data center, dataset, open source, pretrained,
  startup, webpage, website, and Wi-Fi.
- Replace Latinisms with plain English. Use "for example," "that is," "and so
  on," "through," and "compared to."
- Use "refer to" instead of "see," "can" instead of "may" for possibility,
  and "after" instead of "once" for time.
- Do not use "please" in technical instructions.
- Use numerals for specific values, parameters, measurements, and values of 10
  or more. Spell out zero through nine in general prose.
- Include a space between a number and its unit. Use a comma in numbers with
  four or more digits.
- Use title case for headings. Do not style headings with code, bold, italics,
  quotation marks, ampersands, or exclamation marks.
- Use the Oxford comma. Put periods inside quotation marks in U.S. style.
- Use hyphens only for compound modifiers before nouns. Do not hyphenate an
  adverb that ends in "ly."
- Format commands, code, filenames, paths, and API identifiers as code. Use
  bold for UI elements and the greater-than sign for UI navigation.
- Introduce lists, tables, code examples, and images with a complete sentence.
  Use parallel construction in lists.
- Use descriptive link text. Do not use raw URLs in running text or generic
  link text such as "click here" or "read more."
- Write dates as Month DD, YYYY. Omit the year when it matches the publication
  year. Write time with a 12-hour clock and include minutes only when needed.
- Do not rewrite quoted UI labels, API field names, or audience role labels in
  tables to enforce second person.
- Provide useful alt text and preserve a logical heading hierarchy.
- Verify commands, flags, API names, defaults, and technical claims against
  source code or another checked-in source of truth.
- Do not rewrite literal code, identifiers, commands, URLs, or quoted terminal
  and API output to satisfy prose rules.
- Apply rules to improve clarity. Do not make mechanical changes that reduce
  technical accuracy or readability.

## Use DORI for Complete NVIDIA Doc Tools

Follow [NVIDIA DORI Routing](../AGENTS.md#nvidia-dori-routing).
Use the following DORI workflow only when current host capabilities include the
verified NVIDIA documentation Skill Library. Complete the documentation before
the developer opens the pull or merge request.

1. Route the documentation task through DORI. Include the changed source files,
   the user-visible impact, the documentation that might need updates, and the
   required validation.
2. Follow the skill or workflow that DORI returns. Verify product behavior
   against checked-in sources before drafting.
3. When the host supports subagents, start a documentation subagent while the
   primary developer finishes the implementation. Reconcile the documentation
   changes and validation evidence before opening the pull or merge request.
4. When the host does not support subagents, complete the same documentation
   work in the primary task.

If the verified Skill Library is unavailable, inaccessible, or fails, skip DORI.
Do not attempt routing, prompt for setup, or ask for or persist a user
classification. Continue using the Writing Style Guide above.

## Documentation Structure

- Keep `README.md` as the product overview, quick start, example map, and
  documentation index.
- Keep `docs/01-getting-started.md` aligned with the supported Compose recipes,
  prerequisites, and local-development workflow.
- Keep `docs/02-configuration-guide.md` as the index for configuration and
  customization tasks. Put focused procedures in `docs/how-to/`.
- Keep `docs/03-jetson-thor.md` aligned with Jetson-specific Compose and Docker
  sources.
- Keep `docs/04-evaluation-and-performance.md` and the benchmarking-tool
  READMEs aligned with checked-in benchmark procedures and published evidence.
- Use `docs/05-best-practices.md` for production guidance and
  `docs/06-troubleshooting.md` for symptoms, causes, and verified resolutions.
- Keep each `src/examples/<example>/README.md` aligned with that example's
  pipeline, prompts, service catalogs, supported transports, and deployment
  profiles.

## Documentation Sources and Maintenance

- Verify the example list, identifiers, transports, and defaults in
  `examples_registry.yaml`.
- Verify pipeline behavior in `src/examples/<example>/pipeline.py` and shared
  behavior in `src/examples/shared/` and `src/server.py`.
- Verify prompt and service configuration in the example-local `prompts.yaml`,
  `services.cloud.yaml`, and `services.local.yaml` files.
- Verify Compose profiles, service names, ports, image tags, environment
  variables, and hardware placement in `docker-compose.yml`, `docker/`, and
  `.env.example`.
- Verify Python and client dependencies in `pyproject.toml`, `uv.lock`,
  `client/package.json`, and `client/package-lock.json`.
- Preserve the relative-link style used by the existing Markdown files. When
  adding or renaming a page, update `README.md`,
  `docs/02-configuration-guide.md`, and other indexes that should expose it.
- Do not publish secrets, private environment values, unsupported performance
  claims, or results that lack checked-in evidence.

## Documentation Validation

This repository does not configure a dedicated documentation renderer or
Markdown link checker. For documentation-only changes:

1. Run
   `uv run --project . --group dev pre-commit run --files <changed-files>` on
   every changed file.
2. Inspect changed relative links, anchors, commands, and referenced paths.
3. Run `git diff --check`.
4. Record any validation that requires unavailable credentials, NVIDIA
   services, GPUs, or deployment hardware as a gap.

Return the review result, changed documentation paths or rationale, validation
evidence, and agent surface to the primary task for the pull request's
Documentation Writer Review receipt.

When documentation accompanies code, configuration, Compose, or client
changes, also run the applicable validation commands from the root
`AGENTS.md`.
