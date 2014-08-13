#!/usr/bin/env python3
# coding: utf-8

import json
import logging
import resource
import sys

import click

import download
import trainer


@click.command(help="Train model.")
@click.argument("wotstats", type=click.File("rb"))
@click.option("--min-battles", default=10, help="Minimum tank battles.", show_default=True, type=int)
@click.option("--feature-count", default=4, help="Feature count.", show_default=True, type=int)
@click.option("--lambda", default=0.0, help="Regularization parameter.", show_default=True, type=float)
@click.option("--memory-limit", default=6144, help="Maximum RAM in megabytes.", show_default=True, type=int)
def main(wotstats, min_battles, feature_count, memory_limit, **kwargs):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=sys.stderr)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1048576, -1))
    logging.info("Memory limit: %d MiB.", memory_limit)

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
    for i in tank_count:
        row, battles, wins = download.TANK.unpack(wotstats.read(download.TANK.size))
        if battles >= min_battles:
            yield row, battles, wins


def read_model(wotstats, model):
    column = 0
    for i in range(column_count):
        values = list(read_column(wotstats, min_battles))
        if not values:
            continue
        for row, battles, wins in values:
            pass
        column += 1


if __name__ == "__main__":
    main()
