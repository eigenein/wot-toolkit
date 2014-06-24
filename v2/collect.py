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
        data = make_request(session, range(account_id, account_id + LIMIT))
        if not process_data(args.output, data):
            logging.info("Finished on account #%d.", account_id)
            break
        aps = (account_id - args.start) / (time.time() - start_time)
        logging.info("#%d | %.1f a/s | %.1f a/h | %.0f a/d", account_id, aps, aps * 3600.0, aps * 86400.0)


def make_request(session, id_range):
    logging.debug("Making request: %r…", id_range)
    for attempt in range(5):
        if attempt:
            logging.warning("Attempt #%d. Sleeping…", attempt)
            time.sleep(SLEEP_TIME[attempt])
        response = session.get(
            "http://api.worldoftanks.ru/wot/account/tanks/",
            params={
                "application_id": shared.APPLICATION_ID,
                "account_id": ",".join(map(str, id_range)),
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


def process_data(output, data):
    all_null = True
    for account_id, vehicles in data.items():
        account_id = int(account_id)
        if vehicles is None:
            logging.warning("Account #%d: null.", account_id)
            continue
        all_null = False
        output.write(msgpack.packb({
            "account_id": account_id,
            "vehicles": [{
                "tank_id": vehicle["tank_id"],
                "battles": vehicle["statistics"]["battles"],
                "wins": vehicle["statistics"]["wins"],
            } for vehicle in vehicles],
        }))
    return not all_null


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collects user statistics.")
    parser.add_argument("--start", default=1, dest="start", help="start account ID", metavar="<account ID>", type=int)
    parser.add_argument("-o", "--output", help="output file", metavar="<output.msgpack.gz>", required=True, type=shared.GZipFileType("wb"))
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
