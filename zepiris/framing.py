"""Wire format for sending an image pair in one request body.

Shared by the API client and the ML service, and deliberately dependency-free:
the API container does not install the ML extras, so this cannot live beside the
models.

The format is a 4-byte big-endian length, the probe image, then the reference
image filling the rest of the body::

    +--------+------------------+----------------------+
    | uint32 |   probe bytes    |   reference bytes    |
    +--------+------------------+----------------------+

Why not multipart: Starlette spools any part over 1 MB to a temporary **file**,
so a request carrying a normal phone photo writes to disk and reads it back —
pointless I/O for a service that persists nothing, and real disk churn at high
request rates. Framing keeps both images in memory and skips boundary scanning.
"""

from __future__ import annotations

import struct

_HEADER = struct.Struct(">I")

#: Bytes of framing overhead per request, regardless of image size.
HEADER_SIZE = _HEADER.size


class FrameError(ValueError):
    """The body is not a valid image-pair frame."""


def encode_pair_frame(probe: bytes, reference: bytes) -> bytes:
    """Pack two images into one request body."""
    return _HEADER.pack(len(probe)) + probe + reference


def decode_pair_frame(body: bytes) -> tuple[bytes, bytes]:
    """Split a request body into ``(probe, reference)``.

    Raises:
        FrameError: the body is truncated or the declared length does not fit.
    """
    if len(body) < HEADER_SIZE:
        raise FrameError("body too short for frame header")
    (probe_len,) = _HEADER.unpack_from(body, 0)
    end = HEADER_SIZE + probe_len
    if end > len(body):
        raise FrameError(f"declared probe length {probe_len} exceeds body")
    return body[HEADER_SIZE:end], body[end:]
