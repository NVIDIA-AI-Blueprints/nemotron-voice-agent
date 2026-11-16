# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config files and network guidance for the optional phone (SIP) and Redis overlay. Standard library only.

Nothing here runs unless a deployment starts with `run.sh up <profile> --sip`. The network side (router port
forwarding, firewall, the SIP provider) is always done by hand: this module only generates files and prints commands.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Iterable

LIVEKIT_HTTP_PORT = 7880
REDIS_PORT = 6379
SIP_PORT = 5060
RTP_MIN, RTP_MAX = 10000, 10100

# The livekit/sip image reads /sip/config.yaml. Compose mounts the state volume at /sip, so the file has this name.
SIP_CONFIG_FILE = "config.yaml"

E164 = re.compile(r"^\+[1-9]\d{6,14}$")
VOICE_ID = re.compile(r"^[\w.:-]{1,100}$")


def redis_address() -> str:
    """Return the Redis address on the host, for the SIP container (host networking). Loopback only; never exposed."""
    return f"127.0.0.1:{REDIS_PORT}"


def sip_config_yaml(api_key: str, api_secret: str) -> str:
    """Return the config of the livekit/sip container (host networking, so LiveKit and Redis are on loopback)."""
    return (
        f"api_key: {json.dumps(api_key)}\n"
        f"api_secret: {json.dumps(api_secret)}\n"
        f"ws_url: ws://127.0.0.1:{LIVEKIT_HTTP_PORT}\n"
        "redis:\n"
        f"  address: {redis_address()}\n"
        f"sip_port: {SIP_PORT}\n"
        f"rtp_port: {RTP_MIN}-{RTP_MAX}\n"
        "use_external_ip: true\n"  # advertise the public address, not the LAN one, in SIP/SDP
        "logging:\n"
        "  level: info\n"
    )


def parse_cidrs(values: Iterable[str], what: str) -> list[str]:
    """Validate source ranges. An open range (a /0 prefix) is refused: the point of the firewall is to restrict."""
    result = []
    for value in values:
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError:
            raise ValueError(f"{value!r} is not a valid {what} address range (for example 203.0.113.0/24)") from None
        if network.prefixlen == 0:
            raise ValueError(f"{value!r} would allow every address. Give the {what} provider's real ranges")
        result.append(str(network))
    return result


def firewall_commands(
    signaling: list[str], media: list[str], lan_cidr: str, ssh_from: list[str] | None = None
) -> list[str]:
    """Return ufw commands to review and run by hand. Always restricted by source address; never executed here."""
    rtp = f"{RTP_MIN}:{RTP_MAX}"
    commands = [
        "sudo ufw default deny incoming",
        "sudo ufw default allow outgoing",
        f"sudo ufw allow from {lan_cidr} to any port 22 proto tcp   # SSH from your LAN only",
        f"sudo ufw allow from {lan_cidr}   # your LAN: UI and LiveKit",
    ]
    for cidr in ssh_from or []:
        commands.append(f"sudo ufw allow from {cidr} to any port 22 proto tcp   # SSH from another network")
    for cidr in signaling:
        commands += [
            f"sudo ufw allow from {cidr} to any port {SIP_PORT} proto udp",
            f"sudo ufw allow from {cidr} to any port {SIP_PORT} proto tcp",
        ]
    commands += [f"sudo ufw allow from {cidr} to any port {rtp} proto udp" for cidr in media]
    return [*commands, "sudo ufw enable", "sudo ufw status numbered"]


def network_checklist(lan_ip: str, ui_port: int = 3000) -> str:
    """Return the manual network steps with this deployment's ports filled in."""
    return f"""\
Phone (SIP) network setup. These steps are manual: this tool does not touch your router or firewall.

1. Public address. You need a real public IPv4 address (not CGNAT: a router WAN address from 100.64.x.x to
   100.127.x.x means CGNAT). If it changes, update the provider's origination URI and restart livekit-sip, which
   reads its public address at startup. A dynamic DNS name keeps the provider pointed at you, but the restart stays.
2. Router. Give the host a fixed LAN address (now {lan_ip}), turn OFF "SIP ALG" and UPnP, and forward ONLY:
     {SIP_PORT}  UDP+TCP  -> {lan_ip}   (SIP signaling)
     {RTP_MIN}-{RTP_MAX}  UDP  -> {lan_ip}   (RTP audio)
   Never forward 22, {ui_port}, {LIVEKIT_HTTP_PORT}-7882, or {REDIS_PORT}.
3. Host firewall. Router forwards cannot filter by source address, so allow only the provider on the host. Take
   the provider's current signaling and media ranges and print the exact ufw commands (review, then run them):
     run.sh sip-network --lan-ip {lan_ip} --lan <LAN CIDR> --signaling <CIDR> [--signaling <CIDR> ...] --media <CIDR>
   The firewall blocks SSH from every network it does not list. If you SSH in from outside the LAN (a VPN, for
   example), add --ssh-from <CIDR> for it, and keep a second SSH session open while you apply the rules.
4. SIP provider (for example a Twilio Elastic SIP Trunk). Attach your number and add an origination URI
   sip:<your-public-ip-or-ddns-name>:{SIP_PORT};transport=udp .
5. LiveKit. With the deployment running (run.sh up <profile> --sip):
     run.sh sip-setup <profile> --number +15551234567 --scenario bank --allowed-address <SIGNALING CIDR>
6. Check from OUTSIDE your network (a phone hotspot) that {SIP_PORT} is not reachable except from the provider,
   then call the number.
"""
