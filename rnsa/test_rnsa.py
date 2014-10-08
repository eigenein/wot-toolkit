#!/usr/bin/env python3
# coding: utf-8

import math

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
        rnsa.Model(row_count=2, column_count=2, value_count=2, k=2).set_indptr(j, 1)


def test_set_value_positive():
    rnsa.Model(2, 2, 2, 2).set_value(0, 0, 1.0)


def test_set_value_negative():
    with pytest.raises(ValueError):
        rnsa.Model(2, 2, 2, 2).set_value(2, 0, 1.0)


def test_init_centroids():
    model = rnsa.Model(row_count=2, column_count=5, value_count=10, k=10)
    model.init_centroids(-1.0, 1.0)
    assert -1.0 <= model.get_centroid(0)[0] <= 1.0
    assert -1.0 <= model.get_centroid(0)[1] <= 1.0


def test_avg():
    model = rnsa.Model(3, 2, 5, 0)
    model.set_indptr(1, 2)
    model.set_value(0, 0, 2.0)
    model.set_value(1, 1, 4.0)
    model.set_value(2, 0, 1.0)
    model.set_value(3, 1, 7.0)
    model.set_value(4, 2, 4.0)
    assert model._avg(0) == 3.0
    assert model._avg(1) == 4.0


@pytest.mark.parametrize(["j1", "j2", "expected"], [
    (0, 1, 1.0),
    (0, 2, -1.0),
    (0, 3, float("nan")),
    (1, 2, -1.0),
    (1, 3, float("nan")),
    (2, 3, float("nan")),
])
def test_w(j1, j2, expected):
    model = rnsa.Model(row_count=4, column_count=4, value_count=8, k=0)
    model.set_value(0, 0, 0.0)
    model.set_value(1, 1, 2.0)
    model.set_indptr(j=1, index=2)
    model.set_value(2, 0, 0.0)
    model.set_value(3, 1, 2.0)
    model.set_indptr(j=2, index=4)
    model.set_value(4, 0, 0.0)
    model.set_value(5, 1, -2.0)
    model.set_indptr(j=3, index=6)
    model.set_value(6, 2, 1.0)
    model.set_value(7, 3, 2.0)
    w = model._w(j1, j2)
    assert (w == expected) or (math.isnan(expected) and math.isnan(w))


def test_find_nearest_centroid():
    model = rnsa.Model(row_count=2, column_count=2, value_count=4, k=32)
    model.set_value(0, 0, 0.0)
    model.set_value(1, 1, 0.0)
    model.set_indptr(j=1, index=2)
    model.set_value(2, 0, 0.0)
    model.set_value(3, 1, 0.0)
    model.init_centroids(-1.0, 1.0)
    assert model._find_nearest_centroid(0) == model._find_nearest_centroid(1)


def test_step():
    model = rnsa.Model(row_count=2, column_count=5, value_count=10, k=3)
    model.set_value(0, 0, -100.0)
    model.set_value(1, 1, +1.0)
    model.set_indptr(j=1, index=2)
    model.set_value(2, 0, -100.0)
    model.set_value(3, 1, -1.0)
    model.set_indptr(j=2, index=4)
    model.set_value(4, 0, +100.0)
    model.set_value(5, 1, +1.0)
    model.set_indptr(j=3, index=6)
    model.set_value(6, 0, +100.0)
    model.set_value(7, 1, -1.0)
    model.set_indptr(j=4, index=8)
    model.set_value(8, 0, 0.0)
    model.set_value(9, 1, 0.0)
    model.init_centroids(-100.0, 100.0)
    for i in range(100):
        model.step()
        print(model.get_centroid(0), model.get_centroid(1), model.get_centroid(2))
    assert False, (model.get_centroid(0), model.get_centroid(1), model.get_centroid(2))
