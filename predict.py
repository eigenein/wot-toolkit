#!/usr/bin/env python3
# coding: utf-8

import argparse
import json

import numpy
import requests


def main(args):
    print("[INFO] Loading profile.")
    profile = json.load(args.profile)

    print("[INFO] Loading user stats.")
    stats = json.load(args.stats)

    print("[INFO] Get tanks.")
    response = requests.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={"application_id": "demo"})
    response.raise_for_status()
    data = response.json()["data"]
    tank_ids = {value["name"]: tank_id for tank_id, value in data.items()}
    tanks_names = {tank_id: value["name"] for tank_id, value in data.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", dest="profile", help="learned profile", metavar="<profile.json>", required=True, type=argparse.FileType("rt"))
    parser.add_argument(dest="stats", help="user stats", metavar="<user.json>", type=argparse.FileType("rt"))
    main(parser.parse_args())
