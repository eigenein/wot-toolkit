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
@click.option("--memory-limit", default=4096, help="Maximum RAM in megabytes.", show_default=True, type=int)
def main(wotstats, min_battles, memory_limit):
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


if __name__ == "__main__":
    main()
