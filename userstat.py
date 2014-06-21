#!/usr/bin/env python3
# coding: utf-8

import argparse
import json

import requests


def main(args):
    session = requests.Session()

    print("[INFO] Search user.")
    response = session.get("http://api.worldoftanks.ru/wot/account/list/", params={"application_id": "demo", "type": "exact", "search": args.search})
    response.raise_for_status()
    data = response.json()["data"]
    account_id = str(data[0]["account_id"])
    print("[ OK ] Account ID: %s." % account_id)

    print("[INFO] List tanks.")
    response = session.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={"application_id": "demo"})
    response.raise_for_status()
    data = response.json()["data"]
    tanks = {tank_id: value["name"] for tank_id, value in data.items()}
    print("[ OK ] %d tanks." % len(tanks))

    print("[INFO] Get stats.")
    response = session.get("http://api.worldoftanks.ru/wot/account/tanks/", params={"application_id": "demo", "account_id": account_id})
    response.raise_for_status()
    data = response.json()["data"][account_id]
    print("[ OK ] %d tanks." % len(data))

    print("[INFO] Dump ratings.")
    json.dump({tanks[str(value["tank_id"])]: (value["statistics"]["wins"] / value["statistics"]["battles"]) for value in data if value["statistics"]["battles"] >= 50}, args.output, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="search", help="user name", metavar="<search>")
    parser.add_argument("-o", "--output", dest="output", help="output", metavar="<output.json>", type=argparse.FileType("wt"))
    main(parser.parse_args())
