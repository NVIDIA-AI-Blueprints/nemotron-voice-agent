# Getting Started

This guide walks you through deploying the Nemotron Voice Agent on your system.

## Prerequisites

Before you begin, ensure you have the following:

- Access to NVIDIA NGC with valid credentials. Refer to the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).
- Docker with NVIDIA GPU support installed. Refer to the [NIM documentation](https://docs.nvidia.com/nim/riva/asr/latest/getting-started.html#prerequisites).
- NVIDIA API key. Required for accessing NIM ASR, TTS, and LLM models and Docker images. Get yours at [build.nvidia.com](https://build.nvidia.com/).
- Git submodules initialized after cloning. The Docker build requires the `nvidia-pipecat` submodule.

## GPU Requirements

The default local deployment requires **2 NVIDIA GPUs** (Ampere, Hopper, Ada, or later).
- **GPU 0**: For running NVIDIA Nemotron Speech ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.
  - **Total VRAM required for ASR and TTS models: 48 GB**
- **GPU 1**: For running NVIDIA LLM NIM.
  - [Nemotron 3 Nano 30B A3B](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard): 48 GB VRAM
  - [Llama 3.3 Nemotron Super 49B v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/modelcard): 80 GB VRAM

If you use a cloud LLM endpoint instead of the local `nvidia-llm` service, the local deployment only needs the ASR/TTS GPU. See [Switch LLM Models](./how-to/switch-llm-models.md#using-cloud-endpoints).

---

## Deployment Steps

1. Clone the repository and navigate to the root directory of the project.

    ```bash
    git clone https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    cd nemotron-voice-agent
    ```

    If you use GitHub SSH keys, you can use `git@github.com:NVIDIA-AI-Blueprints/nemotron-voice-agent.git` instead.

2. Initialize and update the git submodules.

    ```bash
    git submodule update --init
    ```

3. Configure the environment. To get started, copy the example environment file [config/env.example](../config/env.example) to `.env` in the root directory.

    ```bash
    cp config/env.example .env
    ```

4. Set your NVIDIA API key. For persistent local configuration, add the key to `.env`. For temporary shell-only configuration, export it in your current terminal.

    ```bash
    # Option 1: Add to .env
    NVIDIA_API_KEY=<your-nvidia-api-key>

    # Option 2: Export for this shell session
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    ```

5. Log in to the NVIDIA NGC Docker Registry.

    ```bash
    export NGC_API_KEY=<your-nvidia-api-key>
    docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
    ```

    `NVIDIA_API_KEY` and `NGC_API_KEY` usually use the same value from build.nvidia.com or NGC.

6. Deploy the application.

    ```bash
    docker compose up -d
    ```

    > **Note:** Deployment may take 30-60 minutes on first run while images download and model engines initialize. During this time, some services can show `(health: starting)`.

7. Before you open the UI, enable microphone access in Chrome for the HTTP origin. Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enable **Insecure origins treated as secure**, add `http://<host-ip>:9000` to the list, and restart Chrome.

8. Access the application at `http://<host-ip>:9000/`.

    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.

    ![Nemotron Voice Agent WebRTC UI with microphone controls](./images/ui_webrtc.png)

    To verify the deployment, run `docker compose ps` and confirm the application, ASR, TTS, and LLM services are running or healthy. If services remain unhealthy or stuck in `starting`, see [Troubleshooting](./07-troubleshooting.md).

---

## Optional: Deploy TURN Server for Remote Access

If you need to access the application from remote locations or deploy on cloud platforms, configure a TURN server following these steps.

> **Security:** The following TURN setup is for development and testing. Use strong, unique credentials; restrict inbound traffic to the required ports; and avoid exposing a TURN relay on a shared or production network without additional controls.

The default architecture requires TURN configuration in two places:

- `.env` configures server-side ICE handling for the Python application.
- `frontend/webrtc_ui/src/config.ts` configures browser-side ICE servers and is baked into the UI image at build time.

1. Set environment variables for your public IP address and TURN credentials.

    ```bash
    export HOST_IP_EXTERNAL=<your-public-ip-address>
    export TURN_USERNAME=<strong-turn-username>
    export TURN_PASSWORD=<strong-turn-password>
    ```

2. Deploy the Coturn server.

    ```bash
    docker run -d --network=host instrumentisto/coturn -n --verbose --log-file=stdout \
      --external-ip=$HOST_IP_EXTERNAL --listening-ip=0.0.0.0 --lt-cred-mech --fingerprint \
      --user="${TURN_USERNAME}:${TURN_PASSWORD}" --no-multicast-peers --realm=tokkio.realm.org \
      --min-port=51000 --max-port=52000
    ```

3. Update the `.env` file with TURN server configuration.

    **Important:** Replace `<your-public-ip-address>` with your actual public IP address in the `TURN_SERVER_URL` value below.

    ```bash
    # ----------------------------------------------------------------------------
    # TURN SERVER CREDENTIALS
    # ----------------------------------------------------------------------------

    TURN_SERVER_URL=turn:<your-public-ip-address>:3478
    TURN_USERNAME=<strong-turn-username>
    TURN_PASSWORD=<strong-turn-password>
    ```

4. Update WebRTC UI Configuration in the [webrtc_ui](../frontend/webrtc_ui/src/config.ts) file by replacing the empty `RTC_CONFIG` object with your TURN server configuration.

    **Important:** Replace `<your-public-ip-address>`, `<strong-turn-username>`, and `<strong-turn-password>` with the same values that you used for Coturn and `.env`.

    ```typescript
    // Replace this:
    export const RTC_CONFIG = {};

    // With this:
    export const RTC_CONFIG = {
      iceServers: [
        {
          urls: "turn:<your-public-ip-address>:3478",
          username: "<strong-turn-username>",
          credential: "<strong-turn-password>",
        },
      ],
    };
    ```

    For more information, refer to the [WebRTC TURN Server Documentation](https://webrtc.org/getting-started/turn-server).

5. Restart the Docker Compose services to apply the changes.

    ```bash
    docker compose up --build -d
    ```
