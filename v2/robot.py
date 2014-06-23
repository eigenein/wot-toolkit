#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import json
import pathlib
import logging

import requests

import shared


runtime_info = {
    "version": 0
}


def main(args):
    pass


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
