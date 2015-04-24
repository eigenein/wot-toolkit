#!/usr/bin/env python
# coding: utf-8

"""
Naive prediction algorithm.
"""

import collections
import io
import itertools
import logging
import operator
import sys

import click

import encyclopedia
import kit


@click.command()
@click.argument("input_", type=click.File("rb"))
def main(input_: io.IOBase):
    """
    Run and estimate naive prediction algorithm.
    """
    logging.basicConfig(
        format="%(asctime)s (%(module)s) %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    model = train(input_)
    input_.seek(0)
    precision = estimate(input_, model)

    print_model(model)
    print()
    print("Precision: %.2f." % (100.0 * precision))


def train(input_: io.IOBase):
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


def estimate(input_: io.IOBase, model: dict):
    """Estimates model."""
    true = total = 0
    for i in itertools.count():
        stats = kit.read_account_stats(input_)
        if not stats:
            break
        _, tanks = stats
        # Estimate model for this account.
        account_rating = sum(tank.wins for tank in tanks) / sum(tank.battles for tank in tanks)  # total rating
        recommended_tanks = {tank_id for tank_id, rating in model.items() if rating >= account_rating}
        tanks = {tank.tank_id: tank.wins / tank.battles for tank in tanks}  # mapping from account tank ID to its rating
        estimated_tanks = recommended_tanks & tanks.keys()  # intersection of recommended and account's tanks
        for tank_id in estimated_tanks:
            total += 1
            if tanks[tank_id] >= account_rating:
                true += 1
        # Print statistics.
        if total and i % 100 == 0:
            logging.info(
                "#%d estimate | input: %.1fMiB | precision: %.2f",
                i, input_.tell() / kit.MB, 100.0 * true / total,
            )
    return true / total


def print_model(model: dict):
    """Prints model."""
    for chunk in kit.chop(sorted(model.items(), key=operator.itemgetter(1)), 4):
        for tank_id, rating in chunk:
            print("%16s: %2.2f\t" % (encyclopedia.TANKS[tank_id]["short_name_i18n"], rating * 100.0), end="")
        print()


if __name__ == "__main__":
    main()
