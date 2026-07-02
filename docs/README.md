# Nemotron Voice Agent Docs

Use these docs to deploy, configure, evaluate, and troubleshoot the Nemotron Voice Agent Blueprint. Start with the cloud-only profile if you want the fastest first run, then move to local GPU, DGX Spark, or Jetson Thor profiles when you are ready to self-host models.

## Recommended Reading Order

1. [Getting Started](01-getting-started.md): Choose an example profile, configure `.env`, and start the stack.
2. Example README: Review the example-specific architecture and tunables for the profile you chose.
   - [Generic Cascaded](../src/examples/generic/README.md)
   - [Multilingual Cascaded](../src/examples/multilingual/README.md)
   - [Omni Assistant](../src/examples/omni_assistant/README.md)
   - [Omni Assistant Subagents](../src/examples/omni_assistant_subagents/README.md)
   - [Frontend/Backend Agent](../src/examples/frontend_backend_agent/README.md)
3. [Configuration Guide](02-configuration-guide.md): Find configuration topics by service or feature.
4. [Evaluation and Performance](04-evaluation-and-performance.md): Review benchmark context and reproduction steps.
5. [Troubleshooting](06-troubleshooting.md): Diagnose startup, service, browser, and Jetson issues.

## Production Note

The default Docker Compose deployment is intended for development, demos, and evaluation. Before production use, add authentication, network access controls, managed secrets, valid TLS certificates, retention policies for audio and trace data, readiness probes, and operational runbooks.
