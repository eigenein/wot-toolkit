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

import download
import trainer


METRIC = 10.0


@click.command(help="Train model.")
@click.argument("wotstats", type=click.File("rb"))
@click.option("--min-battles", default=50, help="Minimum tank battles.", show_default=True, type=int)
@click.option("--feature-count", default=16, help="Feature count.", show_default=True, type=int)
@click.option("--lambda", default=0.0, help="Regularization parameter.", show_default=True, type=float)
@click.option("--memory-limit", default=6144, help="Maximum RAM in megabytes.", show_default=True, type=int)
def main(wotstats, min_battles, feature_count, memory_limit, **kwargs):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=sys.stderr)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1048576, -1))
    logging.info("Memory limit: %d MiB.", memory_limit)

    wotstats = io.BufferedReader(wotstats)  # enable buffering

    magic = read_magic(wotstats)
    logging.info("Magic: %s.", magic)
    column_count, value_count = read_header(wotstats)
    logging.info("Columns: %d. Values: %d.", column_count, value_count)

    logging.info("Reading encyclopedia.")
    encyclopedia = read_json(wotstats)
    row_count = len(encyclopedia)
    logging.info("Rows: %d.", row_count)

    logging.info("Initializing model.")
    model = trainer.Model(row_count, column_count, value_count, feature_count, kwargs["lambda"])
    logging.info("Reading model.")
    value_count = read_model(wotstats, min_battles, model)
    logging.info("Value count: %d.", value_count)

    logging.info("Preparing model.")
    model.prepare(0.5)
    logging.info("Initial shuffle.")
    model.shuffle(0, value_count)
    learning_set_size = value_count * 70 // 100
    logging.info("Learning set size: %d.", learning_set_size)

    logging.info("Computing initial RMSE.")
    initial_rmse, _, _, _ = model.step(0, learning_set_size, 0.0, METRIC)
    logging.info("Initial RMSE: %.6f.", initial_rmse)

    logging.info("Starting gradient descent.")
    gradient_descent(model, learning_set_size, initial_rmse)

    _, avg_error, max_error, under_metric = model.step(learning_set_size, value_count, 0.0, METRIC)
    logging.info(
        "Test set: average error - %.9f, maximum error - %.9f, under metric: %.1f%%.",
        avg_error, max_error, 100.0 * under_metric / (value_count - learning_set_size),
    )


def gradient_descent(model, learning_set_size, initial_rmse):
    "Performs gradient descent on model."

    alpha = 0.001
    previous_rmse = initial_rmse

    try:
        for iteration in itertools.count(1):
            model.shuffle(0, learning_set_size)
            rmse, avg_error, max_error, under_metric = model.step(0, learning_set_size, alpha, METRIC)
            if alpha < 1e-08:
                logging.warning("Learning rate is too small. Stopping.")
                break
            logging.info(
                "#%d | a: %.9f | rmse %.9f | d_rmse: %.9f | avg: %.9f | max: %.9f | metric: %.1f%%",
                iteration, alpha, rmse, rmse - previous_rmse, avg_error, max_error, 100.0 * under_metric / learning_set_size,
            )
            alpha *= 1.05 if rmse < previous_rmse else 0.5
            previous_rmse = rmse
    except KeyboardInterrupt:
        pass


def read_magic(wotstats):
    "Reads wotstats magic."
    return wotstats.read(len(download.FILE_MAGIC))


def read_header(wotstats):
    "Reads wotstats header."
    header = wotstats.read(download.FILE_HEADER.size)
    return download.FILE_HEADER.unpack(header)


def read_json(wotstats):
    "Reads JSON from wotstats file."
    length = download.LENGTH.unpack(wotstats.read(download.LENGTH.size))[0]
    logging.info("JSON length: %d B.", length)
    return json.loads(wotstats.read(length).decode("utf-8"))


def read_column(wotstats, min_battles):
    "Reads column from wotstats file."

    magic = wotstats.read(len(download.ACCOUNT_MAGIC))
    if magic != download.ACCOUNT_MAGIC:
        raise ValueError(magic)

    account_id, tank_count = download.ACCOUNT.unpack(wotstats.read(download.ACCOUNT.size))
    for _ in range(tank_count):
        row, battles, wins = download.TANK.unpack(wotstats.read(download.TANK.size))
        if battles >= min_battles:
            yield row, battles, wins


def read_model(wotstats, min_battles, model):
    "Reads model values from wotstats."

    column, value_count = 0, 0
    half_percent = model.column_count // 200
    start_time = time.time()
    # Iterate over columns.
    try:
        for i in range(model.column_count):
            # Progress.
            if i and (i % half_percent == 0):
                percents = i / model.column_count
                eta = int((time.time() - start_time) / percents)
                logging.info(
                    "%4.1f%% | eta: %dm%ds | read: %d | columns: %d | values: %d",
                    100.0 * percents, eta // 60, eta % 60, i, column, value_count,
                )
            # Read column.
            values = list(read_column(wotstats, min_battles))
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
