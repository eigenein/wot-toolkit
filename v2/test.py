#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import collections
import logging

import msgpack


def main(args):
    counter = collections.Counter()
    for i, obj in enumerate(msgpack.Unpacker(args.input)):
        # Check object.
        tank_number, mod = divmod(len(obj), 3)
        assert mod == 1, "invalid object length"
        # Count statistics.
        counter["object_number"] += 1
        counter["tank_number"] += tank_number
        # Log progress.
        position = args.input.tell()
        if i % 100000 == 0:
            logging.info("%d objects | %.1f MiB", i, position / 1048576.0)
    logging.info("Object number: %d.", counter["object_number"])
    logging.info("Tank number: %d.", counter["tank_number"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test statistics file.")
    parser.add_argument(
        dest="input",
        help="input file",
        metavar="<input.msgpack>",
        type=argparse.FileType("rb"),
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
