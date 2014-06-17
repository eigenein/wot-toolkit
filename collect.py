#!/usr/bin/env python3
# coding: utf-8

import argparse
import csv
import itertools
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", required=True, type=file_type)
    parser.add_argument("-b", "--min-battles", default=50, dest="min_battles", type=int)
    args = parser.parse_args()

    session = requests.Session()

    tanks = collect_tanks(session)
    print("Got %d tanks." % len(tanks))

    writer = csv.DictWriter(args.output, fieldnames=["account_id"] + tanks, extrasaction="ignore")
    writer.writeheader()

    start_time = time.time()
    for account_id in itertools.count():
        collect_account(session, account_id, writer, args.min_battles)
        aps = account_id / (time.time() - start_time)
        print("%.1f a/s - %.0f a/h - %.0f a/d" % (aps, aps * 3600.0, aps * 86400.0))


def collect_tanks(session):
    response = session.get("https://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={"application_id": "demo"})
    response.raise_for_status()

    return sorted(response.json()["data"].keys(), key=int)


def collect_account(session, account_id, writer, min_battles):
    response = session.get("https://api.worldoftanks.ru/wot/account/tanks/", params={"application_id": "demo", "account_id": account_id})
    response.raise_for_status()

    data = response.json()["data"][str(account_id)]
    if data is None:
        print("[NULL] %d" % account_id)
        return

    row = {
        str(d["tank_id"]): "%.3f" % (d["statistics"]["wins"] / d["statistics"]["battles"])
        for d in data
        if d["statistics"]["battles"] >= min_battles
    }
    row["account_id"] = account_id

    print("[ OK ] %d" % account_id)
    writer.writerow(row)


def file_type(arg):
    return open(arg, "w", newline="")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
