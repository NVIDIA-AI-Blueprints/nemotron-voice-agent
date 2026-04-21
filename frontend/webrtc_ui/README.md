# Speech to Speech UI

This is a UI for Voice Agent pipeline using Pipecat with WebRTC transport.

## Requirements

- [Node.js](https://nodejs.org/)
- `npm` or a compatible package manager.

All frontend dependencies are listed in `ui/package.json`.

## Using Turn Server

A TURN server is needed for WebRTC connections when clients are behind NATs or firewalls that prevent direct peer-to-peer communication. The TURN server acts as a relay to ensure connectivity in restrictive network environments.

1. Open `src/config.ts`
2. Set `RTC_CONFIG` as below and update Turn Server details

```typescript
export const RTC_CONFIG: ConstructorParameters<typeof RTCPeerConnection>[0] = {
    iceServers: [
      {
        urls: <turn_server_url>,
        username: <turn_server_username>,
        credential: <turn_server_credential>,
      },
    ],
  };
```

3. Save the file and restart the development server


## Run locally

```bash
npm install
npm run dev
```

Then browse to `http://localhost:5173/`.

## Run in production

Create a production build:

```bash
npm install
npm run build
```

This creates an optimized and minified build in `./dist`. The content of this folder can be served as a static website (AWS S3, Github Pages, HTTP server,...). For example:

```bash
python -m http.server --dir dist # serves the build at `localhost:8000`
```

See below for serving the production build on non-localhost origins.

## Run outside localhost

The UI uses the user's microphone. This is only allowed in [secure contexts](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) such as `http://localhost` and `https://<example>`. If the UI is served in a non-secure context (typically: non-https URLs), the UI shows the error `Cannot read properties of undefined (reading 'getUserMedia')`.

Below are a few options to work around this limitation:

- **Allowlist the URL in Chrome**. This is the easiest approach, but not suitable for production, as it requires users to modify their browser settings.
  In Chrome, browse to `chrome://flags/#unsafely-treat-insecure-origin-as-secure` and enable the setting. In the textbox, add the URL origin that serves the UI. For example, if the UI is served at `http://<HOST_IP>:8000/speech`, enter `http://<HOST_IP>:8000`. Then, restart chrome and browse to the UI again.
- **Serve the UI from a hosting provider with HTTPS support**. This is an easy approach that is suitable for production, but requires hosting the UI on a third-party provider.
  Examples:

  - [Github Pages](https://pages.github.com/)
  - [AWS Amplify Hosting](https://docs.aws.amazon.com/amplify/latest/userguide/welcome.html)
  - [Netlify](https://www.netlify.com/)

- **Self-managed webserver with SSL**. This approach is suitable for production and doesn't require a third-party hosting provider, but is harder to implement and manage.
  Examples:
  - [Ubuntu, Apache2 and Let's Encrypt](https://www.digitalocean.com/community/tutorials/how-to-secure-apache-with-let-s-encrypt-on-ubuntu-20-04)
  - [Ubuntu, nginx and Let's Encrypt](https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-20-04)


## Interacting with the UI

- **[Optional]** Click the `Configure` button to:
  - Edit the system prompt (if supported by the backend)
  - Upload custom voice prompts (if using a zero-shot TTS model)
  - Select a TTS voice

- Click the `Start` button to begin interacting.

> **Note:** You can change the TTS voice in real time during an active session. However, the system prompt and zero-shot audio prompts cannot be edited or uploaded while a session is active.
