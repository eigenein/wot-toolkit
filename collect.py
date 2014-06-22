#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import csv
import itertools
import random
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", metavar="<stats.csv>", required=True, type=file_type)
    parser.add_argument("-b", "--min-battles", default=50, dest="min_battles", metavar="<number of battles>", type=int)
    parser.add_argument("--start", default=1, dest="start", metavar="<start ID>", type=int)
    parser.add_argument("--step", default=1, dest="step", metavar="<step>", type=int)
    args = parser.parse_args()

    session = requests.Session()

    tanks = collect_tanks(session)
    print("Got %d tanks." % len(tanks))

    writer = csv.DictWriter(args.output, fieldnames=["account_id"] + tanks, extrasaction="ignore")
    writer.writeheader()

    start_time = time.time()
    tanks_exported = 0

    for account_id in itertools.count(args.start, args.step):
        tanks_exported += collect_account(session, account_id, writer, args.min_battles)
        time_elapsed = time.time() - start_time
        aps = (account_id - args.start) / time_elapsed
        tps = tanks_exported / time_elapsed
        print("%.1f a/s - %.0f a/h - %.0f a/d - %.1f t/h" % (aps, aps * 3600.0, aps * 86400.0, tps * 3600.0))


def collect_tanks(session):
    response = session.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={"application_id": "demo"})
    response.raise_for_status()

    return sorted(response.json()["data"].keys(), key=int)


def collect_account(session, account_id, writer, min_battles):
    while True:

        response = session.get("http://api.worldoftanks.ru/wot/account/tanks/", params={"application_id": "demo", "account_id": account_id})
        if response.status_code == requests.codes.ok:
            break

        print("Retry.")
        time.sleep(random.uniform(0.1, 0.5))

    data = response.json()["data"][str(account_id)]
    if data is None:
        print("[NULL] %d" % account_id)
        return 0

    row = {
        str(d["tank_id"]): "%.3f" % (d["statistics"]["wins"] / d["statistics"]["battles"])
        for d in data
        if d["statistics"]["battles"] >= min_battles
    }
    row["account_id"] = account_id

    print("[ OK ] %d" % account_id)
    writer.writerow(row)

    return len(row) - 1


def file_type(arg):
    return open(arg, "w", newline="")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
