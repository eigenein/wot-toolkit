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


def test_read_uvarints():
    fp = io.BytesIO(b"\x00\x03\x8E\x02")
    assert tuple(kit.read_uvarints(3, fp)) == (0, 3, 270)


def test_write_account_stats():
    account_id, tanks = 3, [
        kit.Tank(270, 86942, 86941),
    ]
    fp = io.BytesIO()
    kit.write_account_stats(account_id, tanks, fp)
    assert fp.getvalue() == b">>\x03\x01\x8E\x02\x9E\xA7\x05\x9D\xA7\x05"


def test_read_account_stats():
    fp = io.BytesIO(b">>\x03\x01\x8E\x02\x9E\xA7\x05\x9D\xA7\x05")
    assert kit.read_account_stats(fp) == (3, [kit.Tank(270, 86942, 86941)])


def test_enumerate_tanks():
    fp = io.BytesIO(b">>\x03\x01\x8E\x02\x9E\xA7\x05\x9D\xA7\x05")
    assert list(kit.enumerate_tanks(fp)) == [kit.AccountTank(3, 270, 86942, 86941)]


def test_enumerate_diff():
    old = [
        kit.AccountTank(1, 1, 10, 5),
        kit.AccountTank(1, 3, 2, 1),
        kit.AccountTank(2, 4, 1, 0),
    ]
    new = [
        kit.AccountTank(1, 2, 12, 6),
        kit.AccountTank(1, 3, 3, 2),
        kit.AccountTank(2, 4, 1, 0),
        kit.AccountTank(2, 5, 1, 0),
    ]
    expected = [
        kit.AccountTank(1, 2, 12, 6),
        kit.AccountTank(1, 3, 1, 1),
        kit.AccountTank(2, 5, 1, 0),
    ]
    assert list(kit.enumerate_diff(old, new)) == expected
