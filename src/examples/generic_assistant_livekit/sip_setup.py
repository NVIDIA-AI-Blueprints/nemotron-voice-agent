# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision the optional phone (SIP) integration. Run through `run.sh sip-setup` and `run.sh sip-network`.

`trunk` creates the LiveKit inbound SIP trunk and dispatch rule for one phone number while the deployment is up with
`--sip`. It is safe to re-run: the trunk and the dispatch rule of a number are updated in place, so changing the use
case or the allowed addresses only needs the same command with new values. The dispatch rule opens a new room per
call, which the agent joins like a browser session. `scenario` and `voice` become participant attributes that the
agent reads.
`network` prints the manual router and firewall steps. It never changes your network.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .credentials import DEFAULT_STATE_DIR, apply_to_environ
from .scenarios import USE_CASES, get_scenario
from .telephony import E164, VOICE_ID, firewall_commands, network_checklist, parse_cidrs


def build_requests(number: str, scenario: str, voice: str | None, allowed: list[str]):
    """Return the inbound trunk, the dispatch rule and the attributes for one number. Pure, so it needs no server."""
    from livekit import api

    if not E164.match(number):
        raise ValueError(f"{number!r} is not an E.164 number (for example +15551234567)")
    scenario_id = get_scenario(scenario).id
    if scenario_id != scenario:
        raise ValueError(f"unknown use case {scenario!r}; choose from {', '.join(USE_CASES)}")
    if voice is not None and not VOICE_ID.match(voice):
        raise ValueError(f"{voice!r} is not a valid voice name")
    attributes = {"scenario": scenario_id, **({"voice": voice} if voice else {})}
    trunk = api.SIPInboundTrunkInfo(
        name=f"inbound-{number}",
        numbers=[number],
        allowed_addresses=allowed,  # an empty list accepts any source; pass the provider's ranges in production
    )
    rule = api.SIPDispatchRuleInfo(
        name=f"call-{number}",  # one rule per number, so a new use case replaces the old one
        rule=api.SIPDispatchRule(dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix="call-")),
        attributes=attributes,
    )
    return trunk, rule, attributes


def find_rule(rules, name: str, trunk_id: str):
    """Return the dispatch rule to update: the one with this name, or any rule already attached to this trunk.

    LiveKit allows one rule per trunk and number, and older versions of this script put the use case in the rule name.
    """
    return next((r for r in rules if r.name == name or trunk_id in r.trunk_ids), None)


def shared_with_others(rule, trunk_id: str) -> bool:
    """Return True when the rule also serves other trunks, or every trunk (an empty list).

    Such a rule is never replaced, because that would change what the other trunks do.
    """
    return not rule.trunk_ids or any(t != trunk_id for t in rule.trunk_ids)


async def run_trunk(args: argparse.Namespace) -> None:
    """Create the trunk and the dispatch rule, or update them when they already exist."""
    from livekit import api

    allowed = parse_cidrs(args.allowed_address, "SIP provider")
    if not allowed:
        print("warning: no --allowed-address given, so the trunk accepts calls from any source.", file=sys.stderr)
    trunk, rule, attributes = build_requests(args.number, args.scenario, args.voice, allowed)
    apply_to_environ(Path(os.environ.get("LIVEKIT_STATE_DIR", DEFAULT_STATE_DIR)))
    lk = api.LiveKitAPI(
        url=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        trunks = (await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())).items
        existing = next((t for t in trunks if t.name == trunk.name), None)
        rules = (await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())).items
        current = find_rule(rules, rule.name, existing.sip_trunk_id if existing else "")
        if current and shared_with_others(current, existing.sip_trunk_id if existing else ""):
            # Replacing it would change what the other trunks do. Nothing has been changed yet.
            raise ValueError(
                f"dispatch rule {current.sip_dispatch_rule_id} ({current.name}) also serves other trunks, so it is "
                "left unchanged. Delete it in LiveKit, or give this number its own rule, and run the command again"
            )
        if existing:
            trunk_id = existing.sip_trunk_id
            await lk.sip.update_inbound_trunk(trunk_id, trunk)
            print(f"updated inbound trunk: {trunk_id} ({args.number})")
        else:
            created = await lk.sip.create_inbound_trunk(api.CreateSIPInboundTrunkRequest(trunk=trunk))
            trunk_id = created.sip_trunk_id
            print(f"created inbound trunk: {trunk_id} ({args.number})")
        rule.trunk_ids.append(trunk_id)
        if current:
            await lk.sip.update_dispatch_rule(current.sip_dispatch_rule_id, rule)
            print(f"updated dispatch rule: {current.sip_dispatch_rule_id} -> {attributes}")
        else:
            created_rule = await lk.sip.create_dispatch_rule(api.CreateSIPDispatchRuleRequest(dispatch_rule=rule))
            print(f"created dispatch rule: {created_rule.sip_dispatch_rule_id} -> {attributes}")
    finally:
        await lk.aclose()


def run_network(args: argparse.Namespace) -> None:
    """Print the checklist and the ufw commands."""
    signaling = parse_cidrs(args.signaling, "signaling")
    media = parse_cidrs(args.media or args.signaling, "media")
    lan = parse_cidrs([args.lan], "LAN")[0]
    ssh_from = parse_cidrs(args.ssh_from, "SSH")
    if not 1 <= args.ssh_port <= 65535:
        raise ValueError(f"{args.ssh_port} is not a valid SSH port")
    print(network_checklist(args.lan_ip, args.ui_port))
    print("ufw commands (review them, then run them yourself):")
    for command in firewall_commands(signaling, media, lan, ssh_from, args.ssh_port):
        print(f"  {command}")


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(prog="sip_setup", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    trunk = sub.add_parser("trunk", help="create the inbound trunk and dispatch rule for one number")
    trunk.add_argument("--number", required=True, help="your phone number in E.164 form, e.g. +15551234567")
    trunk.add_argument("--scenario", default="generic", help=f"use case for this number: {', '.join(USE_CASES)}")
    trunk.add_argument("--voice", help="TTS voice (default: the agent's default)")
    trunk.add_argument(
        "--allowed-address",
        action="append",
        default=[],
        metavar="CIDR",
        help="accept calls only from this source range (repeat). Use your provider's signaling ranges",
    )
    net = sub.add_parser("network", help="print the router and firewall steps (changes nothing)")
    net.add_argument("--lan-ip", required=True, help="this host's fixed LAN address")
    net.add_argument("--lan", required=True, metavar="CIDR", help="your LAN range, e.g. 192.168.1.0/24")
    net.add_argument("--signaling", action="append", required=True, metavar="CIDR", help="provider signaling range")
    net.add_argument("--media", action="append", metavar="CIDR", help="provider media range (default: signaling)")
    net.add_argument(
        "--ssh-from",
        action="append",
        default=[],
        metavar="CIDR",
        help="another network that you SSH in from (a VPN, for example). Repeat. Other SSH sources are blocked",
    )
    net.add_argument("--ssh-port", type=int, default=22, help="the port that sshd listens on (default 22)")
    net.add_argument("--ui-port", type=int, default=3000)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one command."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "trunk":
            asyncio.run(run_trunk(args))
        else:
            run_network(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
