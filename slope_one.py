#!/usr/bin/env python
# coding: utf-8

"""
Slope One collaborative filtering.
"""

import collections


class SlopeOne:
    """
    Slope One implementation.
    """

    def __init__(self):
        self.frequencies = collections.Counter()
        self.diffs = collections.Counter()
        self.counts = collections.Counter()

    def update(self, rated_items: dict):
        """
        Updates model with the user rated items.
        """
        self.counts.update(rated_items.keys())
        for item1 in rated_items:
            for item2 in rated_items:
                if item1 == item2:
                    continue
                self.frequencies[item1, item2] += 1
                self.diffs[item1, item2] += rated_items[item2] - rated_items[item1]

    def predict(self, rated_items: dict, unrated_item):
        """
        Predicts user item rating by the rated items.
        """
        rating_sum, rating_count = 0.0, 0
        for item, rating in rated_items.items():
            rating_sum += (
                (self.counts[item] - 1) *
                (rating + self.diffs[item, unrated_item] / self.frequencies[item, unrated_item])
            )
            rating_count += (self.counts[item] - 1)
        return rating_sum / rating_count
