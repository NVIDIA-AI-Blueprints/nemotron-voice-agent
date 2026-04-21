# Switch LLM Models

The Nemotron Voice Agent supports multiple LLM models with different capabilities and resource requirements. Configure your desired model by editing the [.env](../../config/env.example) file.

## Using a Local LLM NIM Microservice

1. Copy the example configuration [.env](../../config/env.example):

    ```bash
    cp config/env.example .env
    ```

2. Export your NVIDIA API key as an environment variable:

    ```bash
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    ```

3. Edit the `.env` file. The file contains four pre-configured model blocks. To switch models, comment out the current block and uncomment your desired model.

    **Example: Switch from [Nemotron-3-Nano](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard) (default) to [Llama-3.3-Nemotron-Super-49B](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/modelcard)**

    Comment out the default configuration:

    ```bash
    # ----------------------------------------------------------------------------
    # OPTION 1: Nemotron-3-Nano (comment out this block)
    # ----------------------------------------------------------------------------
    # NVIDIA_LLM_IMAGE=nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant
    # NVIDIA_LLM_MODEL=nvidia/nemotron-3-nano
    # TEMPERATURE=1.0
    # TOP_P=1.0
    # ENABLE_THINKING=false
    # MAX_TOKENS=2048
    # NIM_ENABLE_BUDGET_CONTROL=1
    # NIM_ENABLE_KV_CACHE_REUSE=1
    # SYSTEM_PROMPT_SELECTOR=nemotron-3-nano/generic_voice_assistant
    ```

    Uncomment the desired model:

    ```bash
    # ----------------------------------------------------------------------------
    # OPTION 2: Llama-3.3-Nemotron-Super-49B (uncomment this block)
    # ----------------------------------------------------------------------------
    NVIDIA_LLM_IMAGE=nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.15.4
    NVIDIA_LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
    TEMPERATURE=0
    TOP_P=1.0
    NIM_ENABLE_KV_CACHE_REUSE=1
    SYSTEM_PROMPT_SELECTOR=llama-3.3-nemotron-super-49b-v1.5/generic_voice_assistant
    ```

    > **Note:** Each model has a matching `SYSTEM_PROMPT_SELECTOR` value. Use the prompt selector that corresponds to your chosen model. For more information about system prompts, refer to [Customize System Prompts](./customize-system-prompts.md).

4. Restart the services:

    ```bash
    docker compose down
    docker compose up -d
    ```

## Using Cloud Endpoints

Instead of local deployment, you can use NVIDIA's cloud-hosted models on [build.nvidia.com](https://build.nvidia.com/). For example, you can set up the `.env` file to use the [Nemotron-3-Nano](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard) model on build.nvidia.com as follows.

1. Set your NVIDIA API key as an environment variable:
    ```bash
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    ```

2. Update .env with LLM model details
    ```bash
    # In .env file
    NVIDIA_LLM_URL=https://integrate.api.nvidia.com/v1
    NVIDIA_LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b  # Cloud model name
    ```

3. Comment out or remove the `nvidia-llm` service from the [`docker-compose.yml`](../../docker-compose.yml) file.

4. Remove any dependencies on the `nvidia-llm` service in the `python-app` service.
    ```yaml
    # In docker-compose.yml:
    python-app:
      ...
      environment:
      # Docker services endpoints
      # - NVIDIA_LLM_URL=${NVIDIA_LLM_URL:-http://nvidia-llm:8000/v1} <-- comment out or remove this line
      ...
      depends_on:
      # - nvidia-llm  <-- comment out or remove this line
    ```

5. Restart the remaining services:
    ```bash
    docker compose down
    docker compose up -d
    ```

## Selecting Model Profiles in LLM Local Deployment

NVIDIA NIM provides optimized model profiles for different GPU configurations. Each profile defines the runtime engine and optimization criteria for your specific hardware setup.

> **Note:** Before deploying, ensure your system meets the Hardware and Software requirements. For detailed information, refer to the [NVIDIA NIM Getting Started Guide](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#software).

1. Check which model profiles are compatible with your available GPUs using the `list-model-profiles` utility:

    ```bash
    docker run --rm --gpus=all \
      -e NGC_API_KEY=$NVIDIA_API_KEY \
      $NVIDIA_LLM_IMAGE \
      list-model-profiles
    ```

2. To manually select a profile, set the `NIM_MODEL_PROFILE` environment variable in your `.env` file or docker-compose configuration:

    ```bash
    # Using profile name
    NIM_MODEL_PROFILE="tensorrt_llm-A100-fp16-tp1-throughput"

    # Or using profile ID
    NIM_MODEL_PROFILE="751382df4272eafc83f541f364d61b35aed9cce8c7b0c869269cea5a366cd08c"
    ```

    If you do not manually select a profile, NIM automatically chooses the best profile based on the following criteria:

    - Prefers hardware-specific profiles matching your GPU model.
    - Selects backend engines in order: `tensorrt_llm` > `vllm` > `sglang`.
    - Prioritizes more efficient precision formats (for example, `fp8` over `fp16`).
    - Balances latency and throughput optimization targets based on available profiles.

    The selected profile is logged at startup, showing which profile was chosen and why.

For more details on model profiles, refer to the [NVIDIA NIM Model Profiles documentation](https://docs.nvidia.com/nim/large-language-models/latest/profiles.html).
