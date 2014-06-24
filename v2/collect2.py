#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import gzip
import itertools
import logging
import time

import msgpack
import requests

import shared


LIMIT = 100


def main(args):
    session = requests.Session()

    start_time, max_chunk_size = time.time(), 0
    for account_id in itertools.count(args.start, LIMIT):
        data = get_account_tanks(session, range(account_id, account_id + LIMIT))
        chunk_size = save_account_tanks(args.output, data, args.min_battles, args.chunk_align)
        if not chunk_size:
            logging.info("Finished on account #%d.", account_id)
            break
        max_chunk_size = max(max_chunk_size, chunk_size)
        account_number = account_id - args.start + LIMIT
        aps = account_number / (time.time() - start_time)
        logging.info(
            "#%d | %.1f a/s | %.1f a/h | %.0f a/d | %.1fMiB | %d B",
            account_id, aps, aps * 3600.0, aps * 86400.0, args.output.tell() / 1048576.0, max_chunk_size,
        )


def get_account_tanks(session, id_range):
    logging.debug("Get account tanks: %r…", id_range)
    response = session.get(
        "http://api.worldoftanks.ru/wot/account/tanks/",
        params={
            "application_id": shared.APPLICATION_ID,
            "account_id": ",".join(map(str, id_range)),
            "fields": "statistics,tank_id",
        },
    )
    response.raise_for_status()
    payload = response.json()
    return payload["data"]


def save_account_tanks(output, data, min_battles, chunk_align):
    chunk = []
    for account_id, tanks in data.items():
        if tanks is None:
            logging.warning("Account #%s: null.", account_id)
            continue
        all_null, chunk_item = False, []
        for tank in tanks:
            tank_id = tank["tank_id"]
            battles = tank["statistics"]["battles"]
            wins = tank["statistics"]["wins"]
            if battles < min_battles:
                continue
            chunk_item.extend([tank_id, wins, 2 * wins - battles])
        if chunk_item:
            chunk_item.insert(0, int(account_id))
            chunk.append(chunk_item)
    if not chunk:
        return
    payload = gzip.compress(msgpack.packb(chunk))
    assert len(payload) <= chunk_align, "chunk is too large"
    # TODO: write aligned chunk.
    return len(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collects user statistics.")
    parser.add_argument("--start", default=1, dest="start", help="start account ID", metavar="<account ID>", type=int)
    parser.add_argument("-o", "--output", dest="output", help="output file", metavar="<output file>", required=True, type=argparse.FileType("wb"))
    parser.add_argument("--chunk-align", default=20000, dest="chunk_align", help="chunk alignment (default: %(default)s)", metavar="<alignment in bytes>", type=int)
    parser.add_argument("--min-battles", default=50, dest="min_battles", help="minimum number of battles (default: %(default)s)", metavar="<number of battles>", type=int)
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
