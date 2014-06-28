#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging

import msgpack
import requests
import scipy.sparse

import shared


def main(args):
    tank_rows = get_tank_rows()
    logging.info("Tank rows: %d entries.", len(tank_rows))

    logging.info("Make rating matrix.")
    y, r = get_rating_matrix(args.input, tank_rows)


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


def get_rating_matrix(input, tank_rows):
    account_number = 0
    y_data, r_data, indices, indptr = [], [], [], []
    for i, obj in enumerate(msgpack.Unpacker(args.input)):
        account_number += 1
        # Unpack object.
        account_id, *stats = obj
        # Append column.
        indptr.append(len(indices))
        # Append column values.
        for j in range(0, len(stats), 3):
            tank_id, wins, delta = stats[j:j+3]
            battles = 2 * wins - delta
            rating = wins / battles
            # Append values.
            y_data.append(rating)
            r_data.append(1.0)
            indices.append(tank_rows[tank_id])
        # Log progress.
        position = args.input.tell()
        if i % 100000 == 0:
            logging.info("%d objects | %.1f MiB", i, position / 1048576.0)
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer.")
    parser.add_argument(
        dest="input",
        help="input file",
        metavar="<input.msgpack.gz>",
        type=shared.GZipFileType("rb"),
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
