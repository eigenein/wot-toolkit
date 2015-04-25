#!/usr/bin/env python
# coding: utf-8

"""
Slope One collaborative filtering.
"""

import collections
import io
import itertools
import logging
import pickle
import sys

import click

import kit


class SlopeOne:
    """
    Slope One implementation.
    http://www.sbup.com/wiki/Slope_One
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

    def dump(self, fp: io.IOBase):
        pickle.dump((self.frequencies, self.diffs, self.counts), fp)

    def load(self, fp: io.IOBase):
        self.frequencies, self.diffs, self.counts = pickle.load(fp)

    def __str__(self):
        return "%s(counts=<%d items>, frequencies=<%d items>)" % (
            SlopeOne.__name__, len(self.counts), len(self.frequencies))


@click.group()
def main():
    """Run Slope One on statistics file."""
    logging.basicConfig(
        format="%(asctime)s (%(module)s) %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )


@main.command()
@click.argument("input_", type=click.File("rb"))
@click.argument("output", type=click.File("wb"))
def train(input_, output):
    model = SlopeOne()
    logging.info("Training.")
    for i in itertools.count():
        if i % 100 == 0:
            logging.info("#%d | input: %.1fMiB", i, input_.tell() / kit.MB)
        stats = kit.read_account_stats(input_)
        if not stats:
            break
        _, tanks = stats
        model.update({tank.tank_id: tank.wins / tank.battles for tank in tanks})
    logging.info("Trained model %s.", model)
    model.dump(output)
    logging.info("Saved model.")


if __name__ == "__main__":
    main()
