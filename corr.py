#!/usr/bin/env python3
# coding: utf-8

"""
Pearson correlation based recommendations.
"""

import collections
import io
import itertools
import logging
import math
import sys

import click
import requests

import encyclopedia
import kit


def pearson(rated_items_1: dict, rated_items_2: dict) -> float:
    shared_items = set(rated_items_1.keys()) & set(rated_items_2.keys())
    if not shared_items:
        return 0.0
    sum_1 = sum(rated_items_1[item] for item in shared_items)
    sum_2 = sum(rated_items_2[item] for item in shared_items)
    sum_q1 = sum(pow(rated_items_1[item], 2) for item in shared_items)
    sum_q2 = sum(pow(rated_items_2[item], 2) for item in shared_items)
    p_sum = sum(rated_items_1[item] * rated_items_2[item] for item in shared_items)
    n = len(shared_items)
    denominator = math.sqrt((sum_q1 - pow(sum_1, 2) / n) * (sum_q2 - pow(sum_2, 2) / n))
    if denominator < 0.000001:
        return 0.0
    return (p_sum - (sum_1 * sum_2 / n)) / denominator


@click.command()
@click.argument("input_", type=click.File("rb"))
@click.argument("account_id", type=int)
def main(input_: io.IOBase, account_id: int):
    """
    Train and evaluate Pearson based recommendations.
    """
    logging.basicConfig(
        format="%(asctime)s (%(module)s) %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    my_rated_items = {
        int(tank["tank_id"]): tank["statistics"]["wins"] / tank["statistics"]["battles"]
        for tank in requests.get("http://api.worldoftanks.ru/wot/account/tanks/", params={
            "account_id": account_id,
            "application_id": "demo",
            "fields": "tank_id,statistics",
        }).json()["data"][str(account_id)]
    }
    similarity_sums = collections.Counter()
    model = collections.Counter()
    for i in itertools.count():
        if i % 100 == 0:
            logging.info("#%d | input: %.1fMiB", i, input_.tell() / kit.MB)
        stats = kit.read_account_stats(input_)
        if not stats:
            break
        _account_id, tanks = stats
        if _account_id == account_id:
            continue
        other_rated_items = {tank.tank_id: tank.wins / tank.battles for tank in tanks}
        similarity = pearson(my_rated_items, other_rated_items)
        if similarity <= 0.0:
            continue
        for tank_id, rating in other_rated_items.items():
            similarity_sums[tank_id] += similarity
            model[tank_id] += similarity * rating
    print("Model Predictions:")
    for chunk in kit.chop(model.items(), 4):
        for tank_id, rating in chunk:
            print(
                "%16s: %6.2f" % (
                    encyclopedia.TANKS[tank_id]["short_name_i18n"],
                    100.0 * rating / similarity_sums[tank_id],
                ),
                end="",
            )
        print()
    print("My Tanks:")
    for chunk in kit.chop(my_rated_items.items(), 3):
        for tank_id, rating in chunk:
            print(
                "%16s: %6.2f (%5.2f)" % (
                    encyclopedia.TANKS[tank_id]["short_name_i18n"],
                    100.0 * rating,
                    100.0 * (model[tank_id] / similarity_sums[tank_id]),
                ),
                end="",
            )
        print()

if __name__ == "__main__":
    main()
