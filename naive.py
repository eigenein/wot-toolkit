#!/usr/bin/env python
# coding: utf-8

"""
Naive prediction algorithm.
"""

import collections
import itertools
import logging
import operator
import sys

import click

import encyclopedia
import kit


@click.command()
@click.argument("input_", type=click.File("rb"))
def main(input_):
    """
    Run and estimate naive prediction algorithm.
    """
    logging.basicConfig(
        format="%(asctime)s (%(module)s) %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    model = train(input_)
    print_model(model)


def train(input_):
    """Trains model."""
    battles = collections.Counter()
    wins = collections.Counter()
    for i in itertools.count():
        if i % 100 == 0:
            logging.info("#%d training | input: %.1fMiB", i, input_.tell() / kit.MB)
        stats = kit.read_account_stats(input_)
        if not stats:
            break
        _, tanks = stats
        for tank in tanks:
            battles[tank.tank_id] += tank.battles
            wins[tank.tank_id] += tank.wins
    return {tank_id: wins[tank_id] / battles[tank_id] for tank_id in battles}


def print_model(model: dict):
    """Prints model."""
    for chunk in kit.chop(sorted(model.items(), key=operator.itemgetter(1)), 4):
        for tank_id, rating in chunk:
            print("%16s: %2.2f\t" % (encyclopedia.TANKS[tank_id]["short_name_i18n"], rating * 100.0), end="")
        print()


if __name__ == "__main__":
    main()
