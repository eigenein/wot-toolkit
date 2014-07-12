#!/usr/bin/env python3
# coding: utf-8

import argparse
import itertools
import logging
import sys
import time

import numpy
import requests
import scipy.sparse

import planes


def account_counter():
    for i in itertools.count(1, 100):
        yield range(i, i + 100)


def main(args):
    ratings = scipy.sparse.dok_matrix((100000000, len(planes.ALL)), dtype=numpy.float32)
    session = requests.Session()
    account_number = rating_number = 0
    start_time = time.time()
    try:
        for accounts in account_counter():
            account_id = ",".join(map(str, accounts))
            response = session.get("http://api.worldofwarplanes.ru/wowp/account/planes/", params={
                "application_id": args.application_id,
                "fields": "battles,plane_id,wins",
                "account_id": account_id,
            })
            response.raise_for_status()
            data = response.json()["data"]
            for account_stats in data.values():
                if account_stats is None:
                    continue
                empty = True  # TODO
                for plane_stats in account_stats:
                    if plane_stats["battles"] < args.min_battles:
                        continue
                    empty = False
                    ratings[account_number, planes.ALL[plane_stats["plane_id"]]["seq_id"]] = plane_stats["wins"] / plane_stats["battles"]
                    rating_number += 1
                if not empty:  # TODO
                    account_number += 1
            aps = account_number / (time.time() - start_time)
            logging.info(
                "%d-%d | acc.: %d | rat.: %d | rpa: %.1f | %.1f a/s | %.1f a/d",
                accounts.start, accounts.stop, account_number, rating_number, rating_number / account_number, aps, aps * 86400.0
            )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    logging.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect and save account stats.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output file",
        metavar="<stats.mtx>",
        required=True,
        type=argparse.FileType("wb"),
    )
    parser.add_argument(
        "--min-battles",
        default=10,
        dest="min_battles",
        help="minimum number of battles (default: %(default)s)",
        metavar="<number>",
        type=int,
    )
    parser.add_argument(
        "--application-id",
        default="demo",
        dest="application_id",
        help="application ID (default: %(default)s)",
        metavar="<id>",
    )
    parser.add_argument(
        "-l",
        "--log-path",
        default=sys.stderr,
        dest="log",
        help="log file (default: stderr)",
        metavar="<file>",
        type=argparse.FileType("wt"),
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, stream=args.log)
    main(args)
