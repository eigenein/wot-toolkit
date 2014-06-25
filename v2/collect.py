#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

if sys.hexversion < 0x03040000:
    raise ValueError("Python 3.4+ is required.")

import argparse
import concurrent.futures
import datetime
import json
import logging
import time

import msgpack
import requests

import shared


LIMIT = 100
MAX_WORKERS = 4


def main(args, executor):
    session = requests.Session()

    start_time = time.time()
    for account_id in range(args.start, args.end, LIMIT * MAX_WORKERS):
        data = {}
        futures = [
            executor.submit(get_account_tanks, session, range(account_id + i * LIMIT, account_id + i * LIMIT + LIMIT))
            for i in range(MAX_WORKERS)
        ]
        for future in futures:
            data.update(future.result())
        save_account_tanks(args.output, data, args.min_battles)
        account_number = account_id - args.start + LIMIT * MAX_WORKERS
        aps, size = account_number / (time.time() - start_time), args.output.fileobj.tell()
        logging.info(
            "#%d | %.1f a/s | %.1f a/h | %.0f a/d | %.1fMiB | %.0f B/a",
            account_id, aps, aps * 3600.0, aps * 86400.0, size / 1048576.0, size / account_number,
        )

    work_time = time.time() - start_time
    logging.info("Done in %.0fs / %0.1fh.", work_time, work_time / 3600.0)


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


def save_account_tanks(output, data, min_battles):
    all_null = True
    # Order data by account ID.
    data = sorted(
        (int(account_id), vehicles)
        for account_id, vehicles in data.items()
    )
    for account_id, vehicles in data:
        if vehicles is None:
            logging.warning("Account #%d: null.", account_id)
            continue
        all_null, obj = False, [account_id]
        for vehicle in vehicles:
            if vehicle["statistics"]["battles"] < min_battles:
                continue
            obj.extend([
                vehicle["tank_id"],
                vehicle["statistics"]["wins"],
                2 * vehicle["statistics"]["wins"] - vehicle["statistics"]["battles"]],
            )
        msgpack.pack(obj, output)
    return not all_null


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collects user statistics.",
    )
    parser.add_argument(
        "--start",
        default=1,
        dest="start",
        help="start account ID (default: %(default)s)",
        metavar="<account ID>",
        type=int,
    )
    parser.add_argument(
        "--end",
        default=3000,
        dest="end",
        help="end account ID (default: %(default)s)",
        metavar="<account ID>",
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output file",
        metavar="<output.msgpack.gz>",
        required=True,
        type=shared.GZipFileType("wb"),
    )
    parser.add_argument(
        "--min-battles",
        default=50,
        dest="min_battles",
        help="minimum number of battles (default: %(default)s)",
        metavar="<number of battles>",
        type=int,
    )
    args = parser.parse_args()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            main(args, executor)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
