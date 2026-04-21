# Getting Started

This guide walks you through deploying the Nemotron Voice Agent on your system.

## Prerequisites

Before you begin, ensure you have the following:

- Access to NVIDIA NGC with valid credentials. Refer to the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).
- Docker with NVIDIA GPU support installed. Refer to the [NIM documentation](https://docs.nvidia.com/nim/riva/asr/latest/getting-started.html#prerequisites).
- NVIDIA API key. Required for accessing NIM ASR, TTS, and LLM models and Docker images. Get yours at [build.nvidia.com](https://build.nvidia.com/).

## GPU Requirements

This blueprint requires **2 NVIDIA GPUs** (Ampere, Hopper, Ada, or later).
- **GPU 0**: For running NVIDIA Nemotron Speech ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.
  - **Total VRAM required for ASR and TTS models: 48 GB**
- **GPU 1**: For running NVIDIA LLM NIM.
  - [Nemotron 3 Nano 30B A3B](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard): 48 GB VRAM
  - [Llama 3.3 Nemotron Super 49B v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/modelcard): 80 GB VRAM

---

## Deployment Steps

1. Clone the repository and navigate to the root directory of the project.

    ```bash
    git clone git@github.com:NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    cd nemotron-voice-agent
    ```

2. Initialize and update the git submodules.

    ```bash
    git submodule update --init
    ```

3. Configure the environment. To get started, copy the example environment file [.env.example](../config/env.example) to the root directory.

    ```bash
    cp config/env.example .env
    ```

4. Set your NVIDIA API key as an environment variable:

    ```bash
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    ```

5. Log in to the NVIDIA NGC Docker Registry.

    ```bash
    export NGC_API_KEY=<your-nvidia-api-key>
    docker login nvcr.io
    ```

6. Deploy the application.

    ```bash
    docker compose up -d
    ```

    > **Note:** Deployment may take 30-60 minutes on first run.

7. Access the application at `http://<machine-ip>:9000/`

    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.

    ![UI Screenshot](./images/ui_webrtc.png)

    > **Note:** To enable microphone access in Chrome, go to `chrome://flags/`, enable "Insecure origins treated as secure", add `http://<machine-ip>:9000` to the list, and restart Chrome.

    To verify all services are healthy, run `docker compose ps`.

---

## Optional: Deploy TURN Server for Remote Access

If you need to access the application from remote locations or deploy on cloud platforms, configure a TURN server following these steps.

1. Set an environment variable for your public IP address.
    ```bash
    export HOST_IP_EXTERNAL=<your-public-ip-address>
    ```

2. Deploy the Coturn server.

    ```bash
    docker run -d --network=host instrumentisto/coturn -n --verbose --log-file=stdout \
      --external-ip=$HOST_IP_EXTERNAL --listening-ip=0.0.0.0 --lt-cred-mech --fingerprint \
      --user=admin:admin --no-multicast-peers --realm=tokkio.realm.org \
      --min-port=51000 --max-port=52000
    ```

3. Update the `.env` file with TURN server configuration.

    **Important:** Replace `<your-public-ip-address>` with your actual public IP address in the `TURN_SERVER_URL` value below.

    ```bash
    # ----------------------------------------------------------------------------
    # TURN SERVER CREDENTIALS
    # ----------------------------------------------------------------------------

    TURN_SERVER_URL=turn:<your-public-ip-address>:3478
    TURN_USERNAME=admin
    TURN_PASSWORD=admin
    ```

4. Update WebRTC UI Configuration in the [webrtc_ui](../frontend/webrtc_ui/src/config.ts) file by replacing the empty `RTC_CONFIG` object with your TURN server configuration.

    **Important:** Replace `<your-public-ip-address>` with your actual public IP address in the `urls` field below.

    ```typescript
    // Replace this:
    export const RTC_CONFIG = {};

    // With this:
    export const RTC_CONFIG = {
      iceServers: [
        {
          urls: "turn:<your-public-ip-address>:3478",
          username: "admin",
          credential: "admin",
        },
      ],
    };
    ```

    For more information, refer to the [WebRTC TURN Server Documentation](https://webrtc.org/getting-started/turn-server).

5. Restart the Docker Compose services to apply the changes.

    ```bash
    docker compose up --build -d
    ```
