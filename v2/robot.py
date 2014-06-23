#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
import json
import logging
import pathlib
import random
import time

import requests

import shared


LIMIT = 100
SLEEP_TIME = [None, 1.0, 60.0, 600.0, 3600.0]


runtime_info = {
    "account_id": 1,  # start account ID
    "iteration": 1,  # loop iteration number
    "version": 0,  # runtime info version
}


def main(args):
    try:
        run_loop()
    except KeyboardInterrupt:
        logging.warning("Loop is interrupted by user.")


def run_loop():
    logging.info("Running loop…")

    while True:
        logging.info("Starting loop iteration…")

        collect_stats(runtime_info["account_id"])

        logging.info("Loop iteration is finished.")
        runtime_info["iteration"] += 1
        runtime_info["last_account_id"] = runtime_info["account_id"]
        runtime_info["account_id"] = 1


def collect_stats(start):
    session = requests.Session()
    for base_account_id in itertools.count(start, LIMIT):
        runtime_info["account_id"] = base_account_id
        account_ids = range(base_account_id, base_account_id + LIMIT)
        logging.debug("Accounts: %s.", account_ids)
        data = make_request(session, account_ids)
        if data is None:
            continue
        if process_data(data):
            continue
        else:
            break


def make_request(session, account_ids):
    account_ids = ",".join(map(str, account_ids))
    for attempt_number in range(5):
        if SLEEP_TIME[attempt_number] is not None:
            logging.warning("Attempt #%d. Sleeping…")
            time.sleep(SLEEP_TIME[attempt_number])
        response = session.get("http://api.worldoftanks.ru/wot/account/tanks/", params={
            "application_id": shared.APPLICATION_ID,
            "account_id": account_ids,
            "fields": "statistics.wins,statistics.battles,tank_id",
        })
        if response.status_code != requests.codes.ok:
            logging.warning("Status: %d.", response.status_code)
            continue
        payload = response.json()
        if "data" not in payload:
            logging.warning("No data in payload.")
            continue
        return payload["data"]
    logging.error("All attempts failed for accounts %s.", account_ids)
    return None


def process_data(data):
    all_null = True
    data = sorted(data.items())
    for account_id, vehicles in data:
        account_id = int(account_id)
        logging.debug("Account #%d.", account_id)
        runtime_info["account_id"] = account_id
        if vehicles is None:
            logging.warning("Account #%d is null.", account_id)
            continue
        all_null = False
        pass
    return not all_null


def init_runtime_info(args):
    logging.info("Initializing runtime info…")
    if args.runtime_info_path.exists():
        runtime_info.update(json.load(args.runtime_info_path.open("rt")))
    runtime_info["version"] += 1


def save_runtime_info(args):
    logging.info("Saving runtime info…")
    json.dump(runtime_info, args.runtime_info_path.open("wt"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistics robot.")
    parser.add_argument("-r", dest="runtime_info_path", help="runtime info file", metavar="<robot.json>", required=True, type=pathlib.Path)
    args = parser.parse_args()

    init_runtime_info(args)
    try:
        main(args)
    finally:
        save_runtime_info(args)
