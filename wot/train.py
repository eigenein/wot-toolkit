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

import trainer
import wotstats


THRESHOLD = 50.0


@click.command(help="Train model using Collaborative Filtering.")
@click.argument("database", metavar="<database>", type=click.File("rb"))
@click.option("--min-battles", default=50, help="Minimum tank battles.", metavar="<battles>", show_default=True, type=int)
@click.option("--feature-count", default=16, help="Feature count.", metavar="<count>", show_default=True, type=int)
@click.option("--lambda", default=0.0, help="Regularization parameter.", metavar="<lambda>", show_default=True, type=float)
@click.option("--memory-limit", default=6144, help="Maximum RAM in megabytes.", metavar="<mb>", show_default=True, type=int)
def main(database, min_battles, feature_count, memory_limit, **kwargs):
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
    model = trainer.Model(row_count, column_count, value_count, feature_count, kwargs["lambda"])
    logging.info("Reading model.")
    value_count = read_model(database, min_battles, model)
    logging.info("Value count: %d.", value_count)

    logging.info("Preparing model.")
    model.prepare(0.5)
    logging.info("Initial shuffle.")
    model.shuffle(0, value_count)
    learning_set_size = value_count * 70 // 100
    logging.info("Learning set size: %d.", learning_set_size)

    logging.info("Computing initial RMSE.")
    initial_rmse, _, _, _ = model.step(0, learning_set_size, 0.0, THRESHOLD)
    logging.info("Initial RMSE: %.6f.", initial_rmse)

    logging.info("Starting gradient descent.")
    gradient_descent(model, feature_count, learning_set_size, initial_rmse)

    _, avg_error, max_error, precision = model.step(learning_set_size, value_count, 0.0, THRESHOLD)
    logging.info(
        "Test set: average error - %.9f, maximum error - %.9f, precision: %.1f%%.",
        avg_error, max_error, 100.0 * precision,
    )


def gradient_descent(model, feature_count, learning_set_size, initial_rmse):
    "Performs gradient descent on model."

    alpha = 0.001
    previous_rmse = initial_rmse

    try:
        for iteration in itertools.count(1):
            if iteration % feature_count == 0:
                logging.info("Shuffle!")
                model.shuffle(0, learning_set_size)
            rmse, avg_error, max_error, precision = model.step(0, learning_set_size, alpha, THRESHOLD)
            if alpha < 1e-08:
                logging.warning("Learning rate is too small. Stopping.")
                break
            logging.info(
                "#%d | a: %.9f | rmse %.9f | d_rmse: %.9f | avg: %.9f | max: %.9f | precision: %.1f%%",
                iteration, alpha, rmse, rmse - previous_rmse, avg_error, max_error, 100.0 * precision,
            )
            alpha *= 1.10 if rmse < previous_rmse else 0.5
            previous_rmse = rmse
    except KeyboardInterrupt:
        pass


def read_model(database, min_battles, model):
    "Reads model values from database."

    column, value_count = 0, 0
    half_percent = model.column_count // 200
    start_time = time.time()
    # Iterate over columns.
    try:
        for i in range(model.column_count):
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
                model.set_value(value_count, row, column, 100.0 * wins / battles)
                value_count += 1
            column += 1
    except KeyboardInterrupt:
        pass
    # Return actual value count.
    return value_count


if __name__ == "__main__":
    main()
