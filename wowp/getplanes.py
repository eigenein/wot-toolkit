#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging
import pickle

import requests


def main(args):
    r = requests.get("https://api.worldofwarplanes.ru/wowp/encyclopedia/planes/", params={
        "application_id": "demo",
        "fields": "plane_id,name",
    })
    r.raise_for_status()
    data = r.json()["data"]
    planes = {int(plane["plane_id"]): (seq_id, plane["name"]) for seq_id, plane in enumerate(data.values())}
    logging.info("%d planes.", len(planes))
    pickle.dump(planes, args.output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get planes list.")
    parser.add_argument("-o", "--output", help="output file", metavar="<planes.pickle>", required=True, type=argparse.FileType("wb"))
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stderr)
    main(args)
