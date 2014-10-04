#!/usr/bin/env python3
# coding: utf-8

from collections import Counter

import pytest

from diff import Tag, compare_items, compare_files


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
    assert list(compare_items(items1, items2)) == expected


@pytest.mark.parametrize(("accounts1", "accounts2", "expected_accounts", "expected_statistics"), [
    ([(1, [(1, 2, 1)])], [(1, [(1, 2, 1)])], [], Counter({Tag.EQUAL: 1})),
    ([], [(1, [(1, 2, 1)])], [(1, [(1, 2, 1)])], Counter({Tag.NEW: 1})),
    ([(1, [(1, 2, 1)]), (3, [(1, 2, 1)])], [(2, [(1, 2, 1)])], [(2, [(1, 2, 1)])], Counter({Tag.NEW: 1, Tag.DROPPED: 2})),
    ([(1, [(1, 2, 1)])], [], [], Counter({Tag.DROPPED: 1})),
    (
        [(1, [(1, 2, 1)]), (3, [(1, 2, 1)]), (4, [(1, 2, 1)])],
        [(2, [(1, 2, 1)]), (3, [(1, 3, 2)]), (4, [(1, 2, 1)])],
        [(2, [(1, 2, 1)]), (3, [(1, 1, 1)])],
        Counter({Tag.NEW: 1, Tag.CHANGED: 1, Tag.DROPPED: 1, Tag.EQUAL: 1})
    ),
])
def test_compare_files(accounts1, accounts2, expected_accounts, expected_statistics):
    statistics = Counter()
    accounts = list(compare_files(accounts1, accounts2, statistics))
    assert statistics == expected_statistics
    assert accounts == expected_accounts
