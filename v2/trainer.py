#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging

import msgpack

import shared


def main(args):
    counter = collections.Counter()
    for i, obj in enumerate(msgpack.Unpacker(args.input)):
        # Log progress.
        position = args.input.tell()
        if i % 100000 == 0:
            logging.info("%d objects | %.1f MiB", i, position / 1048576.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer.")
    parser.add_argument(
        dest="input",
        help="input file",
        metavar="<input.msgpack.gz>",
        type=shared.GZipFileType("rb"),
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
