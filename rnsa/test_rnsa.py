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


def test_set_indptr_positive():
    rnsa.Model(2, 2, 2, 2).set_indptr(1, 1)


@pytest.mark.parametrize(["j"], [
    (0, ),
    (2, ),
])
def test_set_indptr_negative(j):
    with pytest.raises(ValueError):
        rnsa.Model(2, 2, 2, 2).set_indptr(j, 1)


def test_set_value_positive():
    rnsa.Model(2, 2, 2, 2).set_value(0, 1.0)


def test_set_value_negative():
    with pytest.raises(ValueError):
        rnsa.Model(2, 2, 2, 2).set_value(2, 1.0)


def test_init_centroids():
    model = rnsa.Model(2, 1, 1, 1)
    model.init_centroids(-1.0, 1.0)
    assert -1.0 <= model.get_centroid(0)[0] <= 1.0
    assert -1.0 <= model.get_centroid(0)[1] <= 1.0
