#!/usr/bin/env python3
# coding: utf-8

import difflib
import logging
import sys

import click

import wotstats


@click.command(help="Compare two wotstats files.")
@click.argument("file1", metavar="<file-1.wotstats>", type=click.File("rb"))
@click.argument("file2", metavar="<file-2.wotstats>", type=click.File("rb"))
@click.option("-o", "--output", metavar="<diff.wotstats>", help="Output file.", required=True, type=click.File("wb"))
def main(file1, file2, output):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=sys.stderr)

    account_count1, _, encyclopedia1, accounts1 = wotstats.read(file1)
    accounts1 = enumerate(accounts1)
    account_count2, _, encyclopedia2, accounts2 = wotstats.read(file2)
    accounts2 = enumerate(accounts2)
    
    if encyclopedia1 != encyclopedia2:
        compare_encyclopedias(encyclopedia1, encyclopedia2)
        raise click.ClickException("encyclopedias differ")

    wotstats.write_header(output, 0, 0)

    logging.info("File 1: %d accounts.", account_count1)
    logging.info("File 2: %d accounts.", account_count2)

    output.seek(0)
    wotstats.write_header(output, 0, 0)


def compare_encyclopedias(encyclopedia1, encyclopedia2):
    "Compares two encyclopedias and prints diff."
    matcher = difflib.SequenceMatcher(a=encyclopedia1, b=encyclopedia2, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            print("-%d,%d +%d,%d" % (i1, i2, j1, j2))
        if tag in ("replace", "delete"):
            print("-", encyclopedia1[i1:i2])
        if tag in ("replace", "insert"):
            print("+", encyclopedia2[j1:j2])


if __name__ == "__main__":
    main()
