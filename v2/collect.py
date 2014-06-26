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
import random
import time

import msgpack
import requests

import shared


LIMIT = 100

REQUEST_LIMIT_EXCEEDED = 407
SOURCE_NOT_AVAILABLE = 504


def main(args, executor):
    max_workers = args.max_workers
    http_adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
    session = requests.Session()
    session.mount('http://', http_adapter)

    start_time, real_account_number = time.time(), 0
    for account_id in range(args.start, args.end, LIMIT * max_workers):
        data = {}
        futures = [
            executor.submit(get_account_tanks, session, range(account_id + i * LIMIT, account_id + i * LIMIT + LIMIT))
            for i in range(max_workers)
        ]
        for future in futures:
            data.update(future.result())
        real_account_number += save_account_tanks(args.output, data, args.min_battles)
        account_number = account_id - args.start + LIMIT * max_workers
        aps, size = account_number / (time.time() - start_time), args.output.fileobj.tell()
        logging.info(
            "#%d | %.1f a/s | %.1f a/h | %.0f a/d | %.1fMiB | %.0f B/a",
            account_id, aps, aps * 3600.0, aps * 86400.0, size / 1048576.0, size / account_number,
        )

    work_time = time.time() - start_time
    logging.info("Done in %.0fs / %0.1fh.", work_time, work_time / 3600.0)
    logging.info("Account number: %d.", real_account_number)


def get_account_tanks(session, id_range):
    payload = None
    logging.debug("Get account tanks: %r…", id_range)
    for attempt in range(3):
        # Make API request.
        response = session.get(
            "http://api.worldoftanks.ru/wot/account/tanks/",
            params={
                "application_id": shared.APPLICATION_ID,
                "account_id": ",".join(map(str, id_range)),
                "fields": "statistics,tank_id",
            },
        )
        # Retry if HTTP request is failed.
        if response.status_code != requests.codes.ok:
            logging.warning("Status code: %d.", response.status_code)
            continue
        # Get payload.
        payload = response.json()
        if payload["status"] == "ok":
            return payload["data"]
        # API returned an error.
        logging.warning("Request failed: %r.", payload)
        code = payload["error"]["code"]
        if code == REQUEST_LIMIT_EXCEEDED:
            # Decrease request rate.
            sleep_time = random.random() + 0.1
            logging.warning("Sleeping for %.2fs.", sleep_time)
            time.sleep(sleep_time)
        elif code == SOURCE_NOT_AVAILABLE:
            # Try to repeat request later.
            logging.warning("Sleeping for an hour.")
            time.sleep(3600.0)
    raise ValueError("All attempts failed. Last payload: %r.", payload)


def save_account_tanks(output, data, min_battles):
    real_account_number = 0
    # Order data by account ID.
    data = sorted(
        (int(account_id), vehicles)
        for account_id, vehicles in data.items()
    )
    for account_id, vehicles in data:
        if vehicles is None:
            logging.warning("Account #%d: null.", account_id)
            continue
        real_account_number += 1
        obj = [account_id]
        for vehicle in vehicles:
            if vehicle["statistics"]["battles"] < min_battles:
                continue
            obj.extend([
                vehicle["tank_id"],
                vehicle["statistics"]["wins"],
                2 * vehicle["statistics"]["wins"] - vehicle["statistics"]["battles"]],
            )
        msgpack.pack(obj, output)
    return real_account_number


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
    parser.add_argument(
        "--max-workers",
        default=4,
        dest="max_workers",
        help="maximum number of HTTP workers (default: %(default)s)",
        metavar="<number of workers>",
        type=int,
    )
    args = parser.parse_args()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            main(args, executor)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
