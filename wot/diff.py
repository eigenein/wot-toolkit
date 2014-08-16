#!/usr/bin/env python3
# coding: utf-8

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

    _, _, encyclopedia1, accounts1 = wotstats.read(file1)
    _, _, encyclopedia2, accounts2 = wotstats.read(file2)
    
    if encyclopedia1 != encyclopedia2:
        raise click.ClickException("encyclopedias differ")

    wotstats.write_header(output, 0, 0)

    pass  # TODO: compare accounts

    output.seek(0)
    wotstats.write_header(output, 0, 0)


if __name__ == "__main__":
    main()
