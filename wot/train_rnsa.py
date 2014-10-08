#!/usr/bin/env python3
# coding: utf-8

import io
import itertools
import json
import logging
import resource
import sys
import time

import click

import rnsa
import wotstats


@click.command(help="Train model using k-means.")
@click.argument("database", metavar="<database>", type=click.File("rb"))
@click.option("--min-battles", default=0, help="Minimum tank battles.", metavar="<battles>", show_default=True, type=int)
@click.option("-k", default=1000, help="Cluster count.", metavar="<k>", show_default=True, type=int)
@click.option("--memory-limit", default=6144, help="Maximum RAM in megabytes.", metavar="<mb>", show_default=True, type=int)
def main(database, min_battles, k, memory_limit):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=sys.stderr)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1048576, -1))
    logging.info("Memory limit: %d MiB.", memory_limit)

    database = io.BufferedReader(database)  # enable buffering

    column_count, value_count = wotstats.read_header(database)
    logging.info("Columns: %d. Values: %d.", column_count, value_count)

    logging.info("Reading encyclopedia.")
    encyclopedia = wotstats.read_json(database)
    row_count = len(encyclopedia)
    logging.info("Rows: %d.", row_count)

    logging.info("Initializing model.")
    model = rnsa.Model(row_count, column_count, value_count, k)
    logging.info("Reading model.")
    value_count = read_model(database, min_battles, model)
    logging.info("Value count: %d.", value_count)

    logging.info("Preparing model.")
    model.init_centroids(0.0, 100.0)
    logging.info("Starting k-means.")
    k_means(model)

    logging.info("Reading centroids.")
    nonzero_count = sum(sum(1 for r in model.get_centroid(i) if r != 0.0) for i in range(k))
    logging.info(
        "Centroids: %d nonzero elements of %d (%.1f%%)",
        nonzero_count, row_count * k, 100.0 * nonzero_count / (row_count * k),
    )


def read_model(database, min_battles, model):
    "Reads model values from database."

    column, value_count = 0, 0
    half_percent = model.column_count // 200
    start_time = time.time()
    # Iterate over columns.
    try:
        for i in range(model.column_count):
            model.set_indptr(i, value_count)
            # Progress.
            if i and (i % half_percent == 0):
                progress = i / model.column_count
                eta = int((1.0 - progress) * (time.time() - start_time) / progress)
                logging.info(
                    "%4.1f%% | eta: %dm%02ds | read: %d | columns: %d | values: %d",
                    100.0 * progress, eta // 60, eta % 60, i, column, value_count,
                )
            # Read column.
            account_id, values = wotstats.read_account(database)
            values = [value for value in values if value[1] >= min_battles]
            if not values:
                continue
            # Set column values.
            for row, battles, wins in values:
                model.set_value(value_count, row, 100.0 * wins / battles)
                value_count += 1
            column += 1
    except KeyboardInterrupt:
        pass
    # Return actual value count.
    return value_count


def k_means(model):
    try:
        for iteration in itertools.count(1):
            model.step()
            logging.info("#%d", iteration)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
