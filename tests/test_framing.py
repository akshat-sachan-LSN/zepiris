"""The image-pair wire format: exact bytes out, no expansion, no ambiguity."""

import pytest

from zepiris.framing import (
    HEADER_SIZE,
    FrameError,
    decode_pair_frame,
    encode_pair_frame,
)


@pytest.mark.parametrize(
    ("probe", "reference"),
    [
        (b"\xff\xd8probe", b"\xff\xd8reference"),
        (b"x", b"y"),
        (b"\x00\x00\x00", b"\x00"),  # payloads that look like length headers
        (bytes(range(256)) * 40, bytes(range(256)) * 7),
    ],
)
def test_round_trips_exactly(probe: bytes, reference: bytes) -> None:
    assert decode_pair_frame(encode_pair_frame(probe, reference)) == (probe, reference)


def test_adds_only_the_header_no_encoding_expansion() -> None:
    """The point of framing over base64: bytes cross the wire unexpanded."""
    probe, reference = b"a" * 5000, b"b" * 3000
    frame = encode_pair_frame(probe, reference)
    assert len(frame) == 5000 + 3000 + HEADER_SIZE


def test_empty_sides_survive_the_round_trip() -> None:
    """An empty image is rejected later, by the decoder that can say why."""
    assert decode_pair_frame(encode_pair_frame(b"", b"")) == (b"", b"")


def test_rejects_a_body_too_short_for_a_header() -> None:
    with pytest.raises(FrameError):
        decode_pair_frame(b"\x00\x01")


def test_rejects_a_length_longer_than_the_body() -> None:
    """Must fail rather than read past the end of the buffer."""
    with pytest.raises(FrameError):
        decode_pair_frame((10_000).to_bytes(4, "big") + b"tiny")


def test_boundary_is_the_declared_length_not_a_delimiter() -> None:
    """Image bytes can contain anything, so the split cannot depend on content."""
    probe = encode_pair_frame(b"nested", b"frame")  # a frame inside a payload
    reference = b"\r\n--boundary--\r\n"  # multipart-looking bytes
    assert decode_pair_frame(encode_pair_frame(probe, reference)) == (probe, reference)
