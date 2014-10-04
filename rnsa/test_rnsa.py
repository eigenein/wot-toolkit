#!/usr/bin/env python3
# coding: utf-8

import pytest

import rnsa


def test_init():
    model = rnsa.Model(0, 0)


def test_memory_error():
    with pytest.raises(MemoryError):
        model = rnsa.Model(1000000000, 1000000000)
