#!/usr/bin/env python3
# coding: utf-8

import pytest

import rnsa


def test_init():
    model = rnsa.Model(0, 0, 0, 0)


def test_memory_error():
    with pytest.raises(MemoryError):
        model = rnsa.Model(1000000000, 1000000000, 1000000000, 0)


def test_members():
    model = rnsa.Model(200, 100, 50, 10)
    assert model.row_count == 200
    assert model.column_count == 100
    assert model.value_count == 50
    assert model.k == 10
