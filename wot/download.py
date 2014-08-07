#!/usr/bin/env python3
# coding: utf-8

import argparse
import itertools
import json
import logging
import struct
import sys

import click
import requests


MAGIC = b"WOTSTATS"
HEADER = "=III"


@click.command(help="Download account database.")
@click.option("--application-id", default="demo", help="application ID", show_default=True)
@click.option("-o", "--output", help="output file", required=True, type=click.File("wb"))
@click.option("--log", default=sys.stderr, help="log file", type=click.File("wt"))
def main(application_id, output, log):
    # Initialize logging.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=log)
    # Write empty header.
    write_header(output, 0, 0, True)
    # Download encyclopedia.
    encyclopedia = download_encyclopedia(application_id)
    write_json(output, encyclopedia)
    row_count = len(encyclopedia)
    # Download database.
    column_count, value_count = download_database(output)
    # Seek to the beginning and update header.
    output.seek(0)
    write_header(output, column_count, value_count, False)


def write_header(output, column_count, value_count, is_empty):
    "Writes database header."
    output.write(MAGIC)
    output.write(struct.pack(HEADER, column_count, value_count, 0xDEADBEEF if is_empty else 0))


def download_database(output):
    "Downloads database."
    column_count = value_count = 0
    return column_count, value_count


def download_encyclopedia(application_id):
    "Downloads encyclopedia."
    logging.info("Downloading encyclopedia…")
    response = requests.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={
        "application_id": application_id,
        "fields": "tank_id,name",
    })
    obj = get_response_object(response)
    return [(tank["name"], tank["tank_id"]) for tank in obj["data"].values()]


def get_response_object(response):
    "Gets response object."
    response.raise_for_status()
    obj = response.json()
    if obj["status"] == "error":
        raise ValueError("{0[code]} {0[message]}".format(obj))
    return obj


def write_json(output, obj):
    "Writes serialized object to output."
    s = json.dumps(obj)
    output.write(struct.pack("=i", len(s)))
    output.write(s.encode("utf-8"))


if __name__ == "__main__":
    main()
