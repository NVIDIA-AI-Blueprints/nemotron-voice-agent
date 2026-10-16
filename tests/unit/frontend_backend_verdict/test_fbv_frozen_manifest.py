# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, E501

"""The files that decide FBV's eval (OpenAI Realtime/Tau) behavior are frozen at ``84affa7``.

Plan ``misc/frontend-backend-verdict/fba-removal-plan.md``, section 2.2.1: all of ``core/``, the
Pipecat adapter files no FBA-removal change edits, the test fixtures and the shared test helpers.
A digest may change only with the user's written approval in the PR description.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FBV = ROOT / "src" / "examples" / "frontend_backend_verdict"

#: SHA-256 of each frozen file, recorded at commit 84affa7.
FROZEN = {
    "src/examples/frontend_backend_verdict/adapters/pipecat/client_events.py": "1528232ec8ae62f2ee6ac4cad0d81a16cb643b235b71fa4166e13a3c84968ab4",
    "src/examples/frontend_backend_verdict/adapters/pipecat/frames.py": "691daf42d02927d5b86dc9f0860da7499bf887580c0c5fea973cee6cdb52a634",
    "src/examples/frontend_backend_verdict/adapters/pipecat/serializer.py": "10e811d75a31126398eb1457b1ff2017c4e194ad18b560d2b7dd945475eec011",
    "src/examples/frontend_backend_verdict/adapters/pipecat/services.py": "2ce4f4e0db9dbe8054824b95386802d7fc6e280219ea90cde0cc0cb9de162a24",
    "src/examples/frontend_backend_verdict/adapters/pipecat/session_log.py": "135dc782791601d3d5d2e1fba4ffbbce9347a2d2fd8e391360da614609696835",
    "src/examples/frontend_backend_verdict/adapters/pipecat/settings.py": "d0393ee623f35a1a878f0afca10d48ef38ed53bd900fea9d2a7fbc3390df44d6",
    "src/examples/frontend_backend_verdict/adapters/pipecat/settings/eval.yaml": "664410566684f4f4885a415f56c575376af57ce21fb661e6a2090c12c405fe25",
    "src/examples/frontend_backend_verdict/adapters/pipecat/speech_output.py": "342e8c91ec44ebb6d87bd9bd184fcbeff4cc6427386abeb3ed09c3b6e13f916b",
    "src/examples/frontend_backend_verdict/adapters/pipecat/timing.py": "51b92f3e6f8dd85678d37e70abcfcdba5ed973f6edf9d09cbcc683b03dd6a19a",
    "src/examples/frontend_backend_verdict/adapters/pipecat/turn_bridge.py": "3e29cdf706b8eb66e18d7e911734a462f8a80985ff26e364224ac1099b67a16c",
    "src/examples/frontend_backend_verdict/adapters/pipecat/warmup.py": "cad8b6f4bead23a241f5f4b1696de7728377b6eeacc2a2b6cf91bc105f7a87df",
    "src/examples/frontend_backend_verdict/core/__init__.py": "7924c82ff5f25c7c88bb5386ca4a6eb6a9d671be880261650f20737fc0e77433",
    "src/examples/frontend_backend_verdict/core/agent/__init__.py": "3deba38492691ee589be2f05f9960ec48054f38d1d229fd4b8ba8ceda5726ac2",
    "src/examples/frontend_backend_verdict/core/agent/agent.py": "23f54b3349acd19ec81b59fb5a6653ff1281f5341ff3bf1bacacaca7be2e2c4d",
    "src/examples/frontend_backend_verdict/core/agent/errors.py": "95d085a9e63112a9a67c609602b1c0fd1175833e311f02ef874f1d61da371a2d",
    "src/examples/frontend_backend_verdict/core/agent/events.py": "2b48290ef8268ab4cd2f2662a1e1365a94770dfa69f05feb0701506cd889c829",
    "src/examples/frontend_backend_verdict/core/agent/factory.py": "eb0628a449067698256df877bf8715e87ed8fcc61a40d05493b71234ecc9904c",
    "src/examples/frontend_backend_verdict/core/agent/instructions.py": "c45af92df2a5568304e7a1159fe2a83a7588b06fb237b8fb3bf3d8242da159fc",
    "src/examples/frontend_backend_verdict/core/agent/llm.py": "5be3af520023b0e71769f7923cc924e62a75e7049b0e92720e5a9ef32dd15c9e",
    "src/examples/frontend_backend_verdict/core/agent/messages.py": "07055c3f8c13d3fe693f06fd27411b071c0e8c9cca870737abe782ad238fc78d",
    "src/examples/frontend_backend_verdict/core/agent/prompts.py": "9847702c84ce09a9a4e7cc879a0fb9fba90817381c6b55262b0d199dd6a30e7d",
    "src/examples/frontend_backend_verdict/core/agent/session.py": "7e88e4aed4afa52738cb2ca153c16473bcb513ac5a6549f1eca89bec4d1a5bd3",
    "src/examples/frontend_backend_verdict/core/audio/__init__.py": "d93a864646578790ac6bb3377d26567fa5aa9658f8c1e4075bade3d2b06e4376",
    "src/examples/frontend_backend_verdict/core/audio/formats.py": "25bc8ceebc9357cd223c26fccb85d809614b90383205637dbe967b1a00bcf81e",
    "src/examples/frontend_backend_verdict/core/audio/g711.py": "44e745e6777483f744bc716086a3951aa614dfbd511202ef90fceb1cbdd2b449",
    "src/examples/frontend_backend_verdict/core/audio/pcm.py": "5c6f0128e1f8813f690b41dd0adce695d6b18e6a35533cddcd7ee74ca43893ea",
    "src/examples/frontend_backend_verdict/core/audio/resample.py": "23ea19b35913cb74da3f44c85ba385304eb37eb9fb73dcc8f9c1f15fb5cd4ed3",
    "src/examples/frontend_backend_verdict/core/backend/__init__.py": "2f3289db7b01b22d90b1a5adf85298c27cce1c15349d8dde107497f6069d0cd0",
    "src/examples/frontend_backend_verdict/core/backend/backend.py": "a473adcecde253cca5fecac94572d79f2a9d8e038f7200d2e8d556d8e936130c",
    "src/examples/frontend_backend_verdict/core/backend/context.py": "31ac0abed6cdb9b8d56097a98bfd433736e60aec501e01397192e904062ca1a9",
    "src/examples/frontend_backend_verdict/core/backend/history.py": "f82bef487b668be59ef223b2265a96ed1f93204b7d1d04b3d202a86cfdd8c763",
    "src/examples/frontend_backend_verdict/core/backend/tool_protocol.py": "76fe00d8efd81feffd9aa549e094ff00c482c3f99c1d94ce45ad4a9a53cb9a68",
    "src/examples/frontend_backend_verdict/core/clock.py": "3e33c0d9a65fe3d9e182293185f2ab2624e8042fe7341adda05607678f0779c8",
    "src/examples/frontend_backend_verdict/core/config/__init__.py": "656c42905ea88a61ad0411bd0713ffdac15d4e298f9aa418388d689c208ef281",
    "src/examples/frontend_backend_verdict/core/config/agent.py": "fa6314772d9d1b0589a7c2172d268b4476c19fcf95508bca8e0eff7bf5764337",
    "src/examples/frontend_backend_verdict/core/config/agent.yaml": "455391aaef2c299b1a8aadf423327540f47e5ebe0e8eb2905761f266b47024ca",
    "src/examples/frontend_backend_verdict/core/config/catalog.py": "a85a697f033a11062be7864016f3d45751c09aca9847e12d307d76dec5eff2a4",
    "src/examples/frontend_backend_verdict/core/config/profiles.py": "8520bcf7a09206a35fec87831a5223696f63a5c9d886a3c1e6dd6961b26b32ee",
    "src/examples/frontend_backend_verdict/core/config/profiles/default.yaml": "e33bb939693b03e3db2b986f49c6817761fd511ba2f614f0b778a8cfd94fd73f",
    "src/examples/frontend_backend_verdict/core/config/profiles/eval.yaml": "8d679673df87331f42f1d296c948d87f684f8a5eb25974d68832ff1866da734e",
    "src/examples/frontend_backend_verdict/core/config/prompts/generic.yaml": "a29ee978525777ad9811935c0f3efa1784d948889676ee13642745d4023a4a0b",
    "src/examples/frontend_backend_verdict/core/config/prompts/tau.yaml": "a29ee978525777ad9811935c0f3efa1784d948889676ee13642745d4023a4a0b",
    "src/examples/frontend_backend_verdict/core/config/voice.py": "f0eb171146a07e4d5271ae5a70f14a64bc3524abc8e4f33a36c8f94971d07d71",
    "src/examples/frontend_backend_verdict/core/config/voice_agent.yaml": "2c32d4ae2f1a81628f650811df78128f296cef003200a6bc69e90da11bae6ab5",
    "src/examples/frontend_backend_verdict/core/engine/__init__.py": "ae8952a114b299755380ebe3be0a8bf8e9d9178226b429818672a6bf933a7af2",
    "src/examples/frontend_backend_verdict/core/engine/event_log.py": "097e8f63061d4996bea274839c1566539b465380c5a91a523b777c58a49c9fb2",
    "src/examples/frontend_backend_verdict/core/engine/filler.py": "e50e14e83a8577553d20142eabc37422c137f8c5d5b6288647277309cbbae898",
    "src/examples/frontend_backend_verdict/core/engine/history_repair.py": "ac25aba6a5ab177095c23dca474a9bfbbb9c27a42202f1375f6b60b2d9162037",
    "src/examples/frontend_backend_verdict/core/engine/interfaces.py": "d70b9832d468b7c8072affdcd8757ae5d5d2a5c1f6b8bdb9f370d4c304d58f27",
    "src/examples/frontend_backend_verdict/core/engine/playback.py": "93867b54f005e1bea0ca6a9996ed3eb21223b198ed09d64c232a40f71d1078c4",
    "src/examples/frontend_backend_verdict/core/engine/port.py": "c80776010935d13b50c6ca645ae18e7618a95bb13da33c92cd42d87edef36df8",
    "src/examples/frontend_backend_verdict/core/engine/response_writer.py": "b32a4cdc95b99b2be3884d390af63a3ed91092ab12a7cbb602470ada61694bef",
    "src/examples/frontend_backend_verdict/core/engine/runner.py": "3e6b9ebae5eb2b670cef1a323947541ca2b604b695d8349ff41aa428dc6aebb8",
    "src/examples/frontend_backend_verdict/core/engine/scripted.py": "d1bc6300fc4eea8c9abf81d1517145f4b36b8ad68f034955def878f861c829f1",
    "src/examples/frontend_backend_verdict/core/engine/server_events.py": "9c70cafdc69f5657dae7ff2f7d28aa059bfa5493927c4852ae0fea1555611101",
    "src/examples/frontend_backend_verdict/core/engine/session_settings.py": "419ff0c23fc6647b6f3f1e0b2896841f951c28f13af5d2d4ba17e79fdf3b2151",
    "src/examples/frontend_backend_verdict/core/engine/turn_api.py": "9e2787f83c68cf0f3858cf6cc70a1e1f135b03ba7e376622e96be5d7473bff92",
    "src/examples/frontend_backend_verdict/core/engine/turn_manager.py": "a67f01b409548d89bbe14ef89355225c4acae00687495a6fdb78c9471b6e870e",
    "src/examples/frontend_backend_verdict/core/engine/write_gate.py": "9ef8bf3838a8ceef8f45c7d2c5a778e195de8b8600ca3fbd3a806482d0583da6",
    "src/examples/frontend_backend_verdict/core/errors.py": "740bdf99c761c87a2ff35e6a3485b70986c990904aece1195ae3fe9a5f2c8ae3",
    "src/examples/frontend_backend_verdict/core/frontend/__init__.py": "ec2bd03aefff961e86ac9e6ae194ab3050a98d1e20329a54dbf8bdba1cc64cdf",
    "src/examples/frontend_backend_verdict/core/frontend/delegation.py": "69555bd6f75dee8980a64f7d9a95c936a573a6ac395a4b03e046cf0185fa0991",
    "src/examples/frontend_backend_verdict/core/frontend/frontend.py": "b1c86cf0f4fcae62b6a3ca97633eab99290e1a530609b1a2dff54b880e5b261b",
    "src/examples/frontend_backend_verdict/core/normalization/__init__.py": "9c227be2b4dcf857e79b4b5438005ccc1e9ad5ca382e428dea0f126b6772c986",
    "src/examples/frontend_backend_verdict/core/normalization/arguments.py": "2f1f00630da33a72de8fa57508213cd68f96d927c41c5970b7a81a833a07f6a5",
    "src/examples/frontend_backend_verdict/core/normalization/prompts.py": "5703a21b951507b36940761525da86185ed13ded96945e0074b33c0e98714a4b",
    "src/examples/frontend_backend_verdict/core/normalization/rules.py": "1ecb87663a157101ead95441a78345db60811828129d71b3984ceb1926356279",
    "src/examples/frontend_backend_verdict/core/normalization/schema_rules.py": "0f72a1fa5449e579b05a7293d6011899d7f10b4912cf9d299d51cbb7f8e857fe",
    "src/examples/frontend_backend_verdict/core/normalization/transcript.py": "f937f360e6857e389f7ea101b57bfa4d9de779ff6fb083c99116cbe9b44fea6b",
    "src/examples/frontend_backend_verdict/core/tools/__init__.py": "5b23f1d24b916fb610fac5e3e3e516a511f20cc552698619b041825c6b7ce95e",
    "src/examples/frontend_backend_verdict/core/tools/realtime_tools.py": "ce977b03f6648ba5c265142ffa58c10920338416d420b37ec4118707117f5ea6",
    "src/examples/frontend_backend_verdict/core/tools/tools.py": "0a82bd8cf5826dbf4b42104b3dc2efcd8e33d3a50a77cf44d0519da5ab080c84",
    "tests/unit/frontend_backend_verdict/_fbv_core_fakes.py": "f5f7ba21c32e007f3e4109dcf89388a1d86a3bdaad9cd033d4de576e8c25a168",
    "tests/unit/frontend_backend_verdict/_fbv_demo_tools.py": "761cabb9f325a896e31abe16475914bb52673c67f89fac05e56d4e50c9d55fa7",
    "tests/unit/frontend_backend_verdict/_fbv_fakes.py": "1e63fdeb24baa346667f262290b464973448c92cbb23d99722ca99ef3ed27e61",
    "tests/unit/frontend_backend_verdict/_fbv_golden.py": "f9ead7e31ab1cc528aff7d43b41beea996890d29d0f259282c83d12a2540c31c",
    "tests/unit/frontend_backend_verdict/_fbv_normalization_replay.py": "743c070571666c787a5dd2a593b6f925bc3124c54e5f5769f3193404ea577c1d",
    "tests/unit/frontend_backend_verdict/_fbv_parity.py": "9ede8bea5a7d9b41fb5c84d2ccd0a624fda666901b438aadc7c9ee8cb1888f28",
    "tests/unit/frontend_backend_verdict/_fbv_pipecat_fakes.py": "cb4189892a69a4cdd36964d9fc960c2dc6b14381b6e8e47dee4a79140e5a85d5",
    "tests/unit/frontend_backend_verdict/fixtures/profiles/backend_only.yaml": "1363749cb15936a6d99ba3c04dbaae2569af73cdd5adbd45cb743bbcc73c101e",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/airline__backend+frontend__backend.json": "b47e2ef64da328452d15ebca860e9fb4d728cf80d662209a1c5d1e8cbc3b1de7",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/airline__backend+frontend__frontend.json": "1b7469977a619e805151bafe591884c98084f04ac5e1bd1a0365fb85e0305eea",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/airline__backend__backend.json": "b47e2ef64da328452d15ebca860e9fb4d728cf80d662209a1c5d1e8cbc3b1de7",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/airline__backend__frontend.json": "386a9f7f7a31397a1dcbf5b4917336de3e5fe05c033230c2a14567b9a86d4920",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/mock__backend+frontend__backend.json": "40c96c6e0272797c990ca0bd0d52ec42bc1ef9ca4b0d325b5eb576035e82437e",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/mock__backend+frontend__frontend.json": "de3f3e19ca5c835f1c8875a22878d7388d22d32783e2e2621d0189524380e054",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/mock__backend__backend.json": "40c96c6e0272797c990ca0bd0d52ec42bc1ef9ca4b0d325b5eb576035e82437e",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/mock__backend__frontend.json": "5d2c352624974b8660c6a56b16f10bb3c3f8a68172aa5d5e429d6bdb8e4ed803",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/retail__backend+frontend__backend.json": "36ee73abc56ddccfada236a0b595ad4e46d1857d837a7c88615ef37bcb892977",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/retail__backend+frontend__frontend.json": "0e4aa5864a806b31a9699bd09ef71a3a1ce6f6a57121984351be6d2b58a00eaf",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/retail__backend__backend.json": "36ee73abc56ddccfada236a0b595ad4e46d1857d837a7c88615ef37bcb892977",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/retail__backend__frontend.json": "c27a0f71acf1fa5260cf335ff320125c1b941e592c9118583dc4a33fc7dca8d8",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/telecom__backend+frontend__backend.json": "c3fc0f16b326717c9eeea7a7197846f130c31e3eb30ca0ad60d709ecca68dd8d",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/telecom__backend+frontend__frontend.json": "43a87d7648470938e2f30a1741873857d77bbd0c0c2593a38054ba29e5c19646",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/telecom__backend__backend.json": "c3fc0f16b326717c9eeea7a7197846f130c31e3eb30ca0ad60d709ecca68dd8d",
    "tests/unit/frontend_backend_verdict/fixtures/prompt_snapshots/telecom__backend__frontend.json": "1f91e1f02a78e1fa5f3871bbd13688409d8ca81c1003ef33a126861ef4d4a90b",
    "tests/unit/frontend_backend_verdict/fixtures/tau2_session_update/PROVENANCE.json": "19b4dba5b1d9c8abbb1091871a5a5925dee7ace00b79773b01466b2950d25a3d",
    "tests/unit/frontend_backend_verdict/fixtures/tau2_session_update/airline.json": "596cfb21cc7654df018d700d2c29cb06334c111988c5eeddbb227b0a13a28325",
    "tests/unit/frontend_backend_verdict/fixtures/tau2_session_update/banking_knowledge.json": "9a6c000f72f56d1559dba08d137a9cf2df4b2f8e9be45277a8f8c7296daac772",
    "tests/unit/frontend_backend_verdict/fixtures/tau2_session_update/mock.json": "6a767150f5888881f0d2b59d8eb29e9e27a2d1e61db3cee7cd4a635c49e85ce0",
    "tests/unit/frontend_backend_verdict/fixtures/tau2_session_update/retail.json": "920185012c9dd2fda8643ed142a3ccf76bf8027a5d974666e3ee50c0f6a5ea83",
    "tests/unit/frontend_backend_verdict/fixtures/tau2_session_update/telecom.json": "44c92e3a050d01c8f744fb43565227dcaec00e9b99b1678dbb6e321f8aff5e49",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/configure_and_prompts.json": "2b433cc4b16e31c6ceefe87eeab9d23f9ccc75e71073c7fe16bcd4b822cf801a",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/merged_profile_configs.json": "842c46dd04f70059b4c13b1a08b59f45e73e06ca345688033a5180b2ce65a442",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/pcm_decode.json": "2e1e904bc13bb66092cc92956173636dfe62fa418a1ea31490110895264c880b",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/pipecat_host.json": "d878d077ca839e5d687e19f32f164f8842274ed81cd257e9a473e992c82d4c8a",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_agent_failure.json": "b39a2af7925007a2f2f197c8e100dd1d299ba9f0c0a8cb38749392f3edeb11e9",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_barge_in_after_answer.json": "b9c9bde48c82bb3356c99d9d559c87da07cc5b68c6e55150f89d510040e6abd5",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_barge_in_during_answer.json": "3354bf491a8a79e46a458fa6e6beb250dfa549bccde46c8c56cb53af1b71a948",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_barge_in_filler.json": "d4caef68a240f04517f4ca02428d620b9d9a4e2e5f5335615b8cfd18e8e85b0b",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_filler_then_answer.json": "28b4a663d1394bab9a19ceb115c899bb09c51f9f4e64b87c886f159040918032",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_parallel_tools.json": "c52ecc592ecd0b3c9055b2b35d652c810130b4bd7b47923fbdd9047eaf100175",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_response_cancel.json": "d4ce40618ea9ccce784df94dfe0b037250fcdb2ebaa136ef3abcff8f486764d1",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_review_continue.json": "d832c5b8591a25dbe0145fb608f7ccb7d6d258633523ea7dc70f418f948509e8",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_review_new.json": "8a3ff2c63e694fc9b02f109eaa006048ef1947fa64e2940b031cb230930ac0f4",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_tool_round.json": "bded4bd32776b37f4414178be2b6e792fd875f17e75ec3ea983acbb65aa1f2bd",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_truncate_known_and_unknown.json": "36a079bd0bc6d83d191a1db0c5648154986f4dd45bf4f6de8dfd59d081664e2d",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_tts_failure.json": "64858e70ee7de38c699391571c71620dd6fd07e42c8f6c446f49ebbc72279b35",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_voice_turn.json": "5205646c623950f9470c42d9d028b6acf16c7f3e00d3682154d358b3d4422eee",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_write_gate_cut.json": "62356c76abee05833edc04b8ca72b1f1d2cdbe1cd8bb43b997761e451cbcfcd8",
    "tests/unit/frontend_backend_verdict/fixtures/v1_golden/scenario_write_gate_heard.json": "438f57bf6e13562a17fcfa780acf05e00d51d31a540d84f2fa7a693fd148ba5d",
    "tests/unit/frontend_backend_verdict/fixtures/verdict_cases.jsonl": "37de0a917b05fd34166de475200567a25aa6544e6d5a98a73dae9dfe23704d09",
}

#: Directories whose file list is frozen too: a new file there could change eval behavior.
FROZEN_TREES = (
    "src/examples/frontend_backend_verdict/core",
    "tests/unit/frontend_backend_verdict/fixtures",
)


def _files(tree: str) -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / tree).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


class FrozenManifestTests(unittest.TestCase):
    def test_every_frozen_file_is_unchanged(self) -> None:
        changed = [
            path for path, digest in FROZEN.items() if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest
        ]
        self.assertEqual(changed, [], "frozen files changed (plan section 2.2.1); this needs the user's approval")

    def test_no_file_was_added_or_removed_in_a_frozen_tree(self) -> None:
        for tree in FROZEN_TREES:
            with self.subTest(tree=tree):
                expected = {path for path in FROZEN if path.startswith(tree + "/")}
                self.assertEqual(_files(tree), expected)


if __name__ == "__main__":
    unittest.main()
