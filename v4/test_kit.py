#!/usr/bin/env python3
# coding: utf-8

import io

import pytest

import kit


uvarint_argvalues = [
    (0, b"\x00"),
    (3, b"\x03"),
    (270, b"\x8E\x02"),
    (86942, b"\x9E\xA7\x05"),
]


@pytest.mark.parametrize(("value", "expected"), uvarint_argvalues)
def test_write_uvarint(value, expected):
    fp = io.BytesIO()
    kit.write_uvarint(value, fp)
    assert fp.getvalue() == expected


@pytest.mark.parametrize(("expected", "bytes_"), uvarint_argvalues)
def test_read_uvarint(bytes_, expected):
    value = kit.read_uvarint(io.BytesIO(bytes_))
    assert value == expected
