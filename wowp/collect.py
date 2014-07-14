#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
import logging
import pickle
import struct
import time

import requests


HEADER_FORMAT = "=sssssssssihi"  # ("wowpstats", rows, columns, values)


class Error(Exception):
    pass


def main(args):
    logging.info("Loading plane list…")
    planes = pickle.load(args.planes)
    logging.info("%d planes.", len(planes))

    logging.info("Collecting…")
    args.output.write(b"\x00" * struct.calcsize(HEADER_FORMAT))  # skip header
    rows, values = collect_all(args.application_id, planes, args.min_battles, args.output)

    logging.info("Writing header…")
    logging.info("Rows: %d, values: %s.", rows, values)
    args.output.seek(0)
    args.output.write(struct.pack(HEADER_FORMAT, b"w", b"o", b"w", b"p", b"s", b"t", b"a", b"t", b"s", rows, len(planes), values))

    logging.info("Finished.")


def collect_all(application_id, planes, min_battles, output):
    session = requests.Session()
    all_rows = all_values = 0
    start_time = time.time()
    try:
        for i in itertools.count(1, 100):
            account_id = range(i, i + 100)
            rows, values = collect_users(application_id, planes, min_battles, output, session, account_id)
            all_rows += rows
            all_values += values
            aps = account_id.stop / (time.time() - start_time)
            logging.info(
                "%d-%d | %.2fMiB | %.1f aps | %.0f apd | %d rows | %.1f Bpr | %d val. | %.1f vpa | %.2f rpa",
                account_id.start, account_id.stop, output.tell() / 1048576.0, aps, aps * 86400.0, all_rows, output.tell() / all_rows, all_values, all_values / all_rows, all_rows / account_id.stop,
            )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Error as e:
        logging.fatal("%s", e)
    return all_rows, all_values


def collect_users(application_id, planes, min_battles, output, session, account_id):
    account_id = ",".join(map(str, account_id))
    for i in range(13):
        # Sleep before next attempts.
        if i:
            sleep_time = 2.0 ** i
            logging.warning("Attempt #%d. Sleep %.0fs…", i, sleep_time)
            try:
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                logging.warning("Sleep is interrupted.")
        # Make request.
        response = session.get("https://api.worldofwarplanes.ru/wowp/account/planes/", params={
            "application_id": application_id,
            "fields": "battles,wins,plane_id",
            "account_id": account_id,
        })
        # Check response.
        if response.status_code == requests.codes.ok:
            response = response.json()
            if response["status"] == "ok":
                return collect_planes(response["data"], min_battles, output)
            else:
                logging.warning("API request failed: %r.", response)
        else:
            logging.warning("Status code: %d.", response.status_code)
    # Abort script.
    raise Error("All attempts failed.")


def collect_planes(data, min_battles, output):
    rows = values = 0
    for account_id, planes in data.items():
        if planes is None:
            continue
        account_id = int(account_id)
        row = []
        for plane in planes:
            if plane["battles"] >= min_battles:
                row.append((plane["plane_id"], plane["wins"] / plane["battles"]))
                values += 1
        if not row:
            continue
        rows += 1
        output.write(struct.pack("=ih", account_id, values))
        row = sorted(row)  # sort by plane_id
        for plane_id, rating in row:
            output.write(struct.pack("=hf", plane_id, rating))
    return rows, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect user stats.")
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--min-battles", default=10, help="minimum number of battles (%(default)s)", metavar="<number>", type=int)
    parser.add_argument("-o", "--output", help="output file", metavar="<my.wowpstats>", required=True, type=argparse.FileType("wb"))
    parser.add_argument("--application-id", default="demo", help="application ID (%(default)s)", metavar="<id>")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stderr)
    main(args)
