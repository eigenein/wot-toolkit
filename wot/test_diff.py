#!/usr/bin/env python3
# coding: utf-8

import pytest

import diff


@pytest.mark.parametrize(("items1", "items2", "expected"), [
    ([], [], []),
    ([(1, 1, 1)], [], []),
    ([(1, 1, 1)], [(1, 1, 1)], []),
    ([(1, 1, 1)], [(2, 1, 1)], [(2, 1, 1)]),
    ([(1, 1, 1)], [(1, 2, 1)], [(1, 1, 0)]),
    ([], [(1, 1, 1)], [(1, 1, 1)]),
    ([(1, 1, 1), (2, 1, 1)], [(2, 10, 5)], [(2, 9, 4)]),
])
def test_compare_items(items1, items2, expected):
    assert list(diff.compare_items(items1, items2)) == expected
