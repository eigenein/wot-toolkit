#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging

import msgpack
import requests
import numpy
import scipy.sparse

import shared


def main(args):
    tank_rows = get_tank_rows()
    logging.info("Tank rows: %d entries.", len(tank_rows))

    logging.info("Make rating matrix.")
    y, r = get_rating_matrix(args.input, tank_rows, args.account_number, args.total_tank_number)


def get_tank_rows():
    response = requests.get(
        "http://api.worldoftanks.ru/wot/encyclopedia/tanks/",
        params={
            "application_id": shared.APPLICATION_ID,
            "fields": "tank_id",
        },
    )
    response.raise_for_status()
    tank_ids = sorted(map(int, response.json()["data"]))
    return {tank_id: i for i, tank_id in enumerate(tank_ids)}


def get_rating_matrix(input, tank_rows, account_number, total_tank_number):
    tank_number = len(tank_rows)
    # Initialize arrays.
    y_data = numpy.zeros(total_tank_number)
    r_data = numpy.zeros(total_tank_number, numpy.bool)
    indices = numpy.zeros(total_tank_number, numpy.int)
    indptr = numpy.zeros(account_number + 1, numpy.int)
    # Fill up arrays.
    tank_counter = 0
    for i, obj in enumerate(msgpack.Unpacker(args.input)):
        # Unpack object.
        account_id, *stats = obj
        # Append column.
        indptr[i] = tank_counter
        # Append column values.
        for j in range(0, len(stats), 3):
            tank_id, wins, delta = stats[j:j+3]
            battles = 2 * wins - delta
            rating = wins / battles
            # Append values.
            y_data[tank_counter] = rating
            r_data[tank_counter] = 1
            indices[tank_counter] = tank_rows[tank_id]
            tank_counter += 1
        # Log progress.
        position = args.input.tell()
        if i % 100000 == 0:
            logging.info("%d objects | %.1f MiB", i, position / 1048576.0)
    indptr[account_number] = tank_counter
    # Truncate arrays.
    logging.info("Tank counter: %d.", tank_counter)
    # Convert to matrices.
    y = scipy.sparse.csc_matrix((tank_number, account_number))
    y.data = y_data
    y.indices = indices
    y.indptr = indptr
    logging.info("Y: %r.", y)
    r = scipy.sparse.csc_matrix((tank_number, account_number), dtype=numpy.bool)
    r.data = r_data
    r.indices = indices
    r.indptr = indptr
    logging.info("R: %r.", r)
    return y, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer.")
    parser.add_argument(
        dest="input",
        help="input file",
        metavar="<input.msgpack.gz>",
        type=shared.GZipFileType("rb"),
    )
    parser.add_argument(
        "-a",
        dest="account_number",
        help="exact account number",
        metavar="<number>",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-t",
        dest="total_tank_number",
        help="exact total tank number (across all accounts)",
        metavar="<number>",
        required=True,
        type=int,
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
