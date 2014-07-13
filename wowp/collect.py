#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging
import pickle

import requests


def main(args):
    logging.info("Loading plane list…")
    planes = pickle.load(args.planes)
    logging.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect user stats.")
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--min-battles", default=10, help="minimum number of battles (%(default)s)", metavar="<number>", type=int)
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stderr)
    main(args)
