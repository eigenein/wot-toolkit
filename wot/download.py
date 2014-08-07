#!/usr/bin/env python3
# coding: utf-8

import argparse
import itertools
import struct

import click
import requests


MAGIC = b"WOTSTATS"
HEADER = "=iii"


@click.command(help="Download account database.")
@click.option("--application-id", default="demo", help="application ID", show_default=True)
@click.option("-o", "--output", help="output file", required=True, type=click.File("wb"))
def main(application_id, output):
    write_header(output, 0, 0, 0)  # skip header


def write_header(output, row_count, column_count, value_count):
    "Writes database header."
    output.write(MAGIC)
    output.write(struct.pack(HEADER, row_count, column_count, value_count))


if __name__ == "__main__":
    main()
