#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import json
import logging

import requests

import shared


def main(args):
    response = requests.get(
        "http://api.worldoftanks.ru/wot/encyclopedia/tanks/",
        params={
            "application_id": shared.APPLICATION_ID,
            "fields": "name",
        },
    )
    response.raise_for_status()
    tanks = response.json()["data"]
    tanks = {tank_id: {"name": tank["name"], "row": i} for i, (tank_id, tank) in enumerate(tanks.items())}
    logging.info("Tanks: %d.", len(tanks))
    json.dump(tanks, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query all tanks from Wargaming Public API.")
    parser.add_argument(
        "-o",
        "--output",
        default=sys.stdout,
        dest="output",
        help="output file",
        metavar="<output.json>",
        type=argparse.FileType("wt"),
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    main(args)
