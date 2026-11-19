# Phone Calls (SIP) for the Generic Assistant on LiveKit

This guide connects the example to a phone number or contact-center platform over the standard Session Initiation Protocol (SIP), using the open source [LiveKit SIP](https://docs.livekit.io/sip/) service. A call becomes a LiveKit room, and the same agent answers it with the same models and [sample use cases](README.md#sample-use-cases). Each number can map to a different use case.

SIP is optional and off unless you start a profile with `--sip`, or with Compose and `LIVEKIT_SIP=1` ([step 3](#3-start-the-deployment-with-sip)). It adds Redis and LiveKit SIP, and it listens on port 5060 and a range of UDP ports on the server's host network. Allow only your provider's addresses to reach them. The network setup is always manual, and the example never changes your router or firewall. The steps use Twilio as the worked example, and any SIP provider works the same way. Check them against the current [LiveKit SIP documentation](https://docs.livekit.io/sip/) and your provider's documentation.

## How It Works

SIP sets up and ends phone calls over IP. It carries only call control, and the audio travels separately as Real-time Transport Protocol (RTP) packets. A call reaches the agent through four pieces:

- **The provider** turns the phone call into a SIP call and sends it to your server.
- **LiveKit SIP** (`livekit-sip`) answers the call and joins a LiveKit room as the caller, turning phone audio into the same stream a browser sends.
- **The LiveKit server and Redis** decide what to do. LiveKit SIP asks the server, over Redis, whether the called number is allowed (an inbound **trunk**) and which room and use case to use (a **dispatch rule**). `run.sh sip-setup` creates both.
- **The LiveKit agent** is the same worker that serves the browser. It joins the room, and the ASR, LLM, and TTS pipeline runs unchanged.

Only `livekit-sip` is reachable from the internet, on SIP port 5060 and RTP ports 10000-10100 (UDP), and only from the provider's addresses. Redis, the LiveKit server, the agent, and the UI stay on the server. The models run where the profile puts them. The Cloud profile calls the NVIDIA Cloud (NVCF) APIs, and the Server and Single GPU profiles run them as local containers.

## Requirements

You need the following:

- **A Linux server.** LiveKit SIP uses host networking, which Docker supports only on Linux. On macOS and Windows, `run.sh up --sip` stops with an error.
- **A server the provider can reach.** The provider opens SIP and RTP connections to you from the internet. An SSH tunnel cannot carry them.
- **A SIP provider account** with a number, for example a Twilio trial number (step 1).

Do the following steps in order. Steps 1 and 2 are optional. Skip a step if you already have what it sets up.

## 1. Set Up Twilio (Optional)

Skip this step if you already have a trunk with a number that points at your server, and you know the provider's signaling and media ranges. This step uses [Twilio Elastic SIP Trunking](https://www.twilio.com/docs/sip-trunking). Twilio's screens and IP ranges change. If something here does not work, check Twilio's documentation.

### 1.1 Find Your Public Address

Run this command on the server, from the repository root:

```bash
src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh sip-address
```

It prints the server's LAN address and network, the router's address, and the public address that the internet sees. It fails if the public address is private or in the carrier-grade network address translation (CGNAT) range (`100.64.0.0/10`), because calls cannot reach you then. It changes nothing.

If you use a router, compare the printed public address with the **WAN** or **Internet** address on the router's status page. If they differ, another NAT sits in front of the router, and you need a cloud server or a public address from your ISP.

Home addresses often change. When yours does, calls stop arriving. Update the origination URI below and restart `livekit-sip`, which reads its public address only at startup. A dynamic DNS name in the URI keeps Twilio pointed at you, but you still restart `livekit-sip` after every change.

### 1.2 Create the Trunk and the Number

In the [Twilio Console](https://console.twilio.com), complete these steps:

1. **Create a trunk.** Go to **Elastic SIP Trunking** > **Trunks**, and select **Create new SIP Trunk**. Give it a name, for example `livekit-test`.
2. **Add the origination URI.** On the trunk's **Origination** tab, add `sip:<public-ip>:5060;transport=udp`, with the address from step 1.1. Twilio sends inbound calls there.
3. **Attach a number.** On the **Numbers** tab, add a Twilio number. You call this number, and you pass it to `sip-setup` in E.164 form (for example `+15551234567`).
4. **Use Twilio's IP ranges.** The following table lists the ranges that Twilio published on its [SIP trunking IP addresses](https://www.twilio.com/docs/sip-trunking/ip-addresses) page when this guide was written (last modified March 9, 2026). The firewall and the trunk accept calls only from them.

   | Use | Range |
   | --- | --- |
   | Signaling, North America (Virginia) | `54.172.60.0/30` |
   | Signaling, North America (Oregon) | `54.244.51.0/30` |
   | Signaling, Europe (Ireland) | `54.171.127.192/30` |
   | Signaling, Europe (Frankfurt) | `35.156.191.128/30` |
   | Signaling, Asia-Pacific (Tokyo) | `54.65.63.192/30` |
   | Signaling, Asia-Pacific (Singapore) | `54.169.127.128/30` |
   | Signaling, Asia-Pacific (Sydney) | `54.252.254.64/30` |
   | Signaling, South America (São Paulo) | `177.71.206.192/30` |
   | Media (RTP audio), all regions | `168.86.128.0/18` |

   Using all the signaling ranges works for any region. If calls are rejected or have no audio, compare this table with [Twilio's IP address page](https://www.twilio.com/docs/sip-trunking/ip-addresses), because Twilio can change the ranges.
5. **Skip termination.** This test does not need outbound calls.

Trial accounts have limits. Twilio can play a notice, and you might need to verify the phone that you call from. Check the trial limits first if a call is blocked.

## 2. Set Up the Network and Firewall (Optional)

Skip this step if the SIP and RTP ports already reach the server, and only from the provider's ranges. Do it before step 3, so that port 5060 is never open to the whole internet.

Environments differ, so this is the strategy:

1. **Give the server a fixed address.** Use a DHCP reservation on the router or a static address.
2. **Run a firewall on the server that allows only the provider.** Deny incoming traffic by default, keep SSH reachable from your own network, and allow the traffic in the following table.

   | Traffic | Port | Allow from |
   | --- | --- | --- |
   | SIP signaling | 5060/UDP (add TCP only if the origination URI uses `transport=tcp`) | The provider's signaling ranges |
   | RTP audio | 10000-10100/UDP | The provider's media range |

   The UI, LiveKit, and Redis listen on `127.0.0.1`, so they need no rule. To open the web UI, use the SSH tunnel in [Remote Browser Access](README.md#remote-browser-access) in the README.
3. **Forward the same two ports from the internet to the server.** On a router, add port forwards to the server's LAN address, and turn off its SIP application layer gateway (ALG) and Universal Plug and Play (UPnP) if it has them. In a cloud, open the ports in the security group. A forward cannot filter by source, which is why the server firewall does that. Forward nothing else.

On a server that uses `ufw` (Ubuntu-based systems, including DGX OS), print the commands for your addresses and run them one at a time. Before you enable the firewall, check that an SSH rule covers where you connect from. A second SSH session helps only if it survives, so keep console access to the server as well. The following command prints the rules for the Twilio ranges:

```bash
src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh sip-network \
  --lan-ip <SERVER_LAN_IP> --lan <LAN_CIDR> \
  --signaling 54.172.60.0/30 --signaling 54.244.51.0/30 \
  --signaling 54.171.127.192/30 --signaling 35.156.191.128/30 \
  --signaling 54.65.63.192/30 --signaling 54.169.127.128/30 \
  --signaling 54.252.254.64/30 --signaling 177.71.206.192/30 \
  --media 168.86.128.0/18
```

`sip-address` printed the LAN values. The firewall blocks SSH from every network that it does not list, and enabling it can drop connections that are already open. If you connect from outside the LAN (a cloud server, or a VPN), add `--ssh-from <CIDR>` for that network. Add `--ssh-port <PORT>` if `sshd` does not use port 22. The command changes nothing, and it refuses open ranges such as `0.0.0.0/0`. It also prints TCP 5060 rules that you can skip for a UDP-only trunk. Afterwards, `sudo ufw status numbered` must show a provider range, never `Anywhere`, for 5060 and 10000:10100. `livekit-sip` uses the host network, so the rules apply to it.

For production, or inside a company or cloud environment, ask your network or cloud administrator to apply the same rules with your environment's own tools. Also have them review Transport Layer Security (TLS) and Secure RTP (SRTP), redundancy, and monitoring, which this guide does not cover.

## 3. Start the Deployment with SIP

Start the profile with SIP, and then check it:

```bash
src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh up cloud --sip     # or server, single-gpu
src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh check cloud
```

`up` starts the profile plus Redis (on `127.0.0.1:6379`) and `livekit/sip` (host network). The SIP service advertises your public address (`use_external_ip: true`), which it finds with an outbound request at startup, so the server needs outbound internet access. `check` confirms that Redis and LiveKit SIP are healthy and listening, that Redis is published on loopback only, and that the SIP log has no errors.

The plain Compose equivalent runs from the repository root:

```bash
LIVEKIT_SIP=1 docker compose \
  --profile generic-assistant-livekit --profile generic-assistant-livekit/sip up -d
```

Set `LIVEKIT_SIP=1` on every `up`, or put it in `.env`. It makes the `livekit-init` service add Redis to the LiveKit server configuration. Without it, the `/sip` profile still starts Redis and LiveKit SIP, but the LiveKit server does not use Redis, and `sip-setup` fails with `sip not connected (redis required)`. For the Server or Single GPU profile, replace `generic-assistant-livekit` with `generic-assistant-livekit/server` or `generic-assistant-livekit/single-gpu`, and keep the `/sip` profile. The `run.sh up <profile> --sip` command sets the variable and the profiles for you. To switch a running stack to SIP with Compose, stop it first with the matching `down`:

```bash
docker compose --profile generic-assistant-livekit --profile generic-assistant-livekit/sip down
```

## 4. Map a Number to a Use Case

The following command creates records inside LiveKit. It does not touch your provider, router, or firewall.

```bash
src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh sip-setup cloud \
  --number +15551234567 --scenario generic \
  --allowed-address 54.172.60.0/30 --allowed-address 54.244.51.0/30 \
  --allowed-address 54.171.127.192/30 --allowed-address 35.156.191.128/30 \
  --allowed-address 54.65.63.192/30 --allowed-address 54.169.127.128/30 \
  --allowed-address 54.252.254.64/30 --allowed-address 177.71.206.192/30
```

Use your number, and pass each signaling range from step 1.2 as `--allowed-address`. The command creates an inbound trunk for the number and a dispatch rule that opens a new room per call. Without `--allowed-address`, the trunk accepts any source, and the command warns.

`--scenario` is `generic`, `insurance`, `clinic`, or `bank`, and `--voice` selects a TTS voice. To change the use case or the allowed addresses of a number, run the command again with new values. Run it with another number for another line.

## 5. Test It

Test the setup with these steps:

1. Wait for the agent with `run.sh wait cloud`.
2. Follow the logs in separate terminals:

   ```bash
   src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh logs cloud sip      # INVITE received, trunk matched
   src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh logs cloud livekit  # room created, participant joined
   src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh logs cloud agent    # tool calls, errors
   ```

3. Call the number from your phone (a verified phone on a trial account).
   After a number is mapped, the web UI also lists it under **Or call the agent**, so people who open the UI know that they can call in.
4. Expect an incoming call in the SIP log, a room named `call-...` in the LiveKit log, and then the greeting. Phone audio is 8 kHz, so recognition and voice quality are lower than in the browser. That is normal.

If a call fails, open the Twilio Console's **Monitor** logs for it. A SIP response code there shows whether the call reached your server.

To confirm that the server does not answer everyone, run this command from a different network, such as a phone hotspot, and not from the server:

```bash
nc -vz -w 5 <public-ip> 5060    # should time out
```

A timeout means that TCP 5060 is closed. You cannot test UDP this way, so check `ufw status` and confirm that the SIP log never shows calls from outside the provider's ranges.

### Clean Up After Testing

Stop the profile with this command:

```bash
src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh down cloud
```

The trunk and dispatch rule records survive restarts in a Docker volume. To delete them, run `run.sh clean <profile>` instead, and confirm the prompt. It also removes the stored LiveKit credentials, so the next start generates new ones unless you set `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in `.env`. Then remove the router's port forwards, and delete the firewall rules that you added. In Twilio, detach or release the number and delete the trunk, so the origination URI does not keep sending calls to an address that is no longer yours.

## Security

A SIP port that accepts any source is scanned and abused within minutes. Restrict it in the layers that the following table lists.

| Layer | What it restricts | How |
| --- | --- | --- |
| Server firewall or cloud security group | 5060 and the RTP range 10000-10100 (UDP) | The commands from `run.sh sip-network` ([step 2](#2-set-up-the-network-and-firewall-optional)). This layer matters most. |
| LiveKit inbound trunk | Which sources can start a call | `--allowed-address` on `run.sh sip-setup` ([step 4](#4-map-a-number-to-a-use-case)) |
| Router | Nothing by source | A forward cannot filter by source, so keep the server firewall |

Review these points before you expose the SIP service:

- **Allow signaling and media separately.** Providers publish separate ranges, and the media range is often larger. Review the list when you change region or trunk, or when calls start failing from a new source. For another provider, use its own list.
- **Match the transport.** A trunk with `transport=udp` needs only UDP on 5060. Add TCP if the provider uses it.
- **A provider with no fixed addresses** cannot be allowlisted. Use TLS with credentials or mutual TLS, or put the server behind a VPN or private link.
- **IPv6.** The example opens IPv4 rules. Add IPv6 rules if the provider uses IPv6.
- **Watch for scans.** Consider a non-standard SIP port, and watch the SIP log for repeated rejected INVITEs.
- **Limit spend.** Set geographic permissions and a spend limit with your provider, and protect outbound trunks with credentials and an IP access list.
- **Encrypt signaling.** Use TLS (`wss://`) for LiveKit signaling when it is reachable from the internet.
- **Protect callers.** Tell callers if you record or keep transcripts. Keep the sample use cases away from real accounts. They use fictional data and a demo PIN.

## Troubleshooting

The following table lists common symptoms and their likely causes.

| Symptom | Likely cause |
| --- | --- |
| The provider reports a failed call and the SIP log is empty | The provider cannot reach 5060. Check the router forward, the origination URI and its `transport`, the server firewall (is the provider's current range allowed?), and CGNAT |
| Calls worked, then stopped arriving, and the SIP log is empty | Your public address probably changed. Run `run.sh sip-address`, update the origination URI, and restart the stack |
| The SIP log shows an incoming call, then `status 486` with `reason "no-dispatch"`, or a 403, a 404, or "no trunk found" | The call reached your server, but LiveKit has no trunk or rule for the called number. Run `sip-setup` with the exact number from the log's `toUser`, in E.164 form. If you set `--allowed-address`, check that the log's `fromIP` is in one of those ranges |
| The call connects but has no audio, or one-way audio | RTP is blocked, or the wrong address is advertised. Open the RTP range to the provider's media range, and turn off the router's SIP ALG |
| The call connects, then silence, then it hangs up | The agent is not registered, or it cannot reach the models. Check the agent log |
| `sip-setup` fails with `sip not connected (redis required)` | The stack started without `LIVEKIT_SIP=1`, so the LiveKit server has no Redis. Stop the stack, then start it again with `run.sh up <profile> --sip`, or with `LIVEKIT_SIP=1` set for Compose ([step 3](#3-start-the-deployment-with-sip)) |
| `up` fails with `port is already allocated` for 6379 or 3000 | Containers of an earlier deployment still hold the ports. Stop that deployment from its own folder with `docker compose --profile <profile> down`. List the holders with `docker ps --format '{{.Names}}\t{{.Ports}}'` |
| `up --sip` stops with "only on Linux" | SIP needs host networking. Use a Linux server |
| It works in the browser but never by phone | The SSH tunnel does not carry phone calls. Follow [step 2](#2-set-up-the-network-and-firewall-optional) |

## Other SIP Peers

The steps apply to any peer that can send calls to a SIP endpoint. Such peers include trunk providers, contact-center platforms (for example Genesys Cloud, Five9, and Amazon Connect), and a company private branch exchange (PBX) or session border controller (SBC). Adapt the steps as follows:

- **Provider setup.** Create a trunk or external SIP connection on the platform that points at your server's address and port 5060.
- **Allow the platform's addresses** in every layer that [Security](#security) lists.
- **Match the called number.** Pass the number that the platform sends as `--number`, in E.164 form. `sip-setup` does not accept a bare PBX extension such as `1001`, so have the platform send a full number.
- **Encryption.** Enterprise platforms often require TLS for signaling and SRTP for media. This example configures neither.
- **Network.** A cloud platform needs a server reachable from the internet. An on-premises platform on the same network needs only a private route, with nothing forwarded on a router.
- **Handing off to a person.** LiveKit supports SIP call transfer, but this example's agent has no handoff tool.

To test first, use a trial trunk or a SIP softphone on the same network. Then switch the peer, and change only the trunk address, the allowed addresses, and the number.
