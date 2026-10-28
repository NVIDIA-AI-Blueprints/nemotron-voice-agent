# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep structured output on the StreamingInput chunks vLLM resumes a request with.

vLLM compiles each chunk's grammar, such as ``response_format`` or a required
tool call, but a resumed request takes only the chunk's sampling params and
keeps decoding without the grammar. ``keep_structured_output`` carries the
chunk's grammar onto the resumed request. The scheduler runs in the engine core
process, which imports this module again when it is spawned instead of forked.

Checked against vLLM 0.31.0 and 0.32.0. Remove it once vLLM's
``StreamingUpdate`` carries ``structured_output_request``.
"""

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus, StreamingUpdate


def keep_structured_output() -> None:
    """Patch vLLM so a resumed StreamingInput request decodes with its new chunk's grammar."""
    from_request = StreamingUpdate.from_request.__func__
    update_request_as_session = Scheduler._update_request_as_session

    def from_request_with_grammar(cls, request):
        update = from_request(cls, request)
        if update is not None:
            update.structured_output_request = request.structured_output_request
        return update

    def update_request_as_session_with_grammar(self, session, update):
        update_request_as_session(self, session, update)
        session.structured_output_request = getattr(update, "structured_output_request", None)
        if session.structured_output_request is not None:
            session.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    StreamingUpdate.from_request = classmethod(from_request_with_grammar)
    Scheduler._update_request_as_session = update_request_as_session_with_grammar
