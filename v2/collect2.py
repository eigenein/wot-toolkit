#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import datetime
import itertools
import json
import logging
import time

import msgpack
import requests

import shared


LIMIT = 100
SLEEP_TIME = [None, 1.0, 60.0, 600.0, 3600.0]


def main(args):
    session = requests.Session()

    start_time = time.time()
    for account_id in itertools.count(args.start, LIMIT):
        data = get_account_tanks(session, range(account_id, account_id + LIMIT))
        if not save_account_tanks(args.output, data, args.min_battles):
            logging.info("Finished on account #%d.", account_id)
            break
        account_number = account_id - args.start + LIMIT
        aps, size = account_number / (time.time() - start_time), args.output.fileobj.tell()
        logging.info(
            "#%d | %.1f a/s | %.1f a/h | %.0f a/d | %.1fMiB | %.0f B/a",
            account_id, aps, aps * 3600.0, aps * 86400.0, size / 1048576.0, size / account_number,
        )


def get_account_tanks(session, id_range):
    logging.debug("Get account tanks: %r…", id_range)
    for attempt in range(5):
        if attempt:
            logging.warning("Attempt #%d. Sleeping…", attempt)
            time.sleep(SLEEP_TIME[attempt])
        response = session.get(
            "http://api.worldoftanks.ru/wot/account/tanks/",
            params={
                "application_id": shared.APPLICATION_ID,
                "account_id": ",".join(map(str, id_range)),
                "fields": "statistics,tank_id",
            },
        )
        if response.status_code != requests.codes.ok:
            logging.warning("Status code: %d.", response.status_code)
            continue
        payload = response.json()
        if "data" not in payload:
            logging.warning("No data.")
            continue
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
    parser = argparse.ArgumentParser(description="Collects user statistics.")
    parser.add_argument("--start", default=1, dest="start", help="start account ID", metavar="<account ID>", type=int)
    parser.add_argument("-o", "--output", dest="output", help="output file", metavar="<output.msgpack.gz>", required=True, type=shared.GZipFileType("wb"))
    parser.add_argument("--min-battles", default=50, dest="min_battles", help="minimum number of battles (default: %(default)s)", metavar="<number of battles>", type=int)
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
