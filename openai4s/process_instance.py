"""The one id for this daemon process.

Kernel background receipts and the server's session-recovery owner import
this module. The kernel does not import the server to learn which process
it is. One run therefore writes the same id into both places.
"""

from __future__ import annotations

import uuid

#: Stable for the life of the process. A restarted daemon draws a new value,
#: so a receipt or a kernel generation from the previous process no longer
#: matches.
PROCESS_INSTANCE_ID = f"daemon-{uuid.uuid4()}"

__all__ = ["PROCESS_INSTANCE_ID"]
