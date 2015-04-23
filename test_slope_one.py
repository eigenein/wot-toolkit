#!/usr/bin/env python
# coding: utf-8

import slope_one


def test_train_predict():
    model = slope_one.SlopeOne()
    model.update({1: 5.0, 2: 3.0, 3: 2.0})
    model.update({1: 3.0, 2: 4.0})
    model.update({2: 2.0, 3: 5.0})
    assert 4.333333 < model.predict({2: 2.0, 3: 5.0}, 1) < 4.333334
