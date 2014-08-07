#!/usr/bin/env python3
# coding: utf-8

import argparse
import itertools
import json
import logging
import operator
import struct
import sys
import time

import click
import requests


FILE_MAGIC = b"WOTSTATS"
ACCOUNT_MAGIC = b"$$";
# Struct instances.
TANK_COUNT = UINT16 = struct.Struct("<H")
ACCOUNT_ID = LENGTH = UINT32 = struct.Struct("<I")
FILE_HEADER = struct.Struct("<III")
TANK = struct.Struct("<HII")


@click.command(help="Download account database.")
@click.option("--application-id", default="demo", help="application ID", show_default=True)
@click.option("-o", "--output", help="output file", required=True, type=click.File("wb"))
@click.option("--log", default=sys.stderr, help="log file", type=click.File("wt"))
def main(application_id, output, log):
    # Initialize logging.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=log)
    # Write empty header.
    write_header(output, 0, 0, True)
    # Download encyclopedia.
    encyclopedia = download_encyclopedia(application_id)
    write_json(output, encyclopedia)
    row_count = len(encyclopedia)
    # Download database.
    column_count, value_count = download_database(application_id, encyclopedia, output)
    # Seek to the beginning and update header.
    output.seek(0)
    write_header(output, column_count, value_count, False)


def write_header(output, column_count, value_count, is_empty):
    "Writes database header."
    logging.info("Writing header…")
    output.write(FILE_MAGIC)
    output.write(FILE_HEADER.pack(column_count, value_count, 0xDEADBEEF if is_empty else 0))


def download_database(application_id, encyclopedia, output):
    "Downloads database."
    logging.info("Starting download…")
    # Reverse encyclopedia.
    reverse_encyclopedia = {tank[1]: row for row, tank in enumerate(encyclopedia)}
    # Initialize statistics.
    column_count = value_count = 0
    start_time = time.time()
    # Prepare session.
    session = requests.Session()
    # Iterate over all accounts.
    try:
        for i in itertools.count(1, 100):
            # Make request.
            sequence = ",".join(map(str, range(i, i + 100)))
            obj = get_account_tanks(session, application_id, sequence)
            # Iterate over accounts.
            for account_id, tanks in obj["data"].items():
                if tanks is None:
                    continue
                tanks = [tank for tank in tanks if tank["statistics"]["battles"] >= 10]
                if not tanks:
                    continue
                value_count += write_column(int(account_id), tanks, reverse_encyclopedia, output)
                column_count += 1
            # Print statistics.
            apd = 86400.0 * column_count / (time.time() - start_time)
            logging.info(
                "#%d | %d acc. | apd: %.1f | %d val. | %.1fMiB",
                i, column_count, apd, value_count, output.tell() / 1048576.0,
            )
    except KeyboardInterrupt:
        logging.warning("Interrupted.")
    except:
        logging.exception("Fatal error.")
    return column_count, value_count


def get_account_tanks(session, application_id, sequence):
    "Requests tanks for the specified account ID sequence."
    for attempt in range(10):
        if attempt:
            time.sleep(5.0)  # sleep on next retries
        try:
            return get_response_object(session.get("http://api.worldoftanks.ru/wot/account/tanks/", params={
                "application_id": application_id,
                "account_id": sequence,
                "fields": "statistics,tank_id",
            }))
        except KeyboardInterrupt:
            raise
        except:
            logging.exception("Can't get account tanks.")
    raise ValueError("all attempts failed")


def write_column(account_id, tanks, reverse_encyclopedia, output):
    "Writes account column."
    output.write(ACCOUNT_MAGIC)
    output.write(ACCOUNT_ID.pack(account_id))
    output.write(TANK_COUNT.pack(len(tanks)))
    tanks = sorted(tanks, key=operator.itemgetter("tank_id"))
    for tank in tanks:
        output.write(TANK.pack(
            reverse_encyclopedia[tank["tank_id"]],
            tank["statistics"]["battles"],
            tank["statistics"]["wins"],
        ))
    return len(tanks)


def download_encyclopedia(application_id):
    "Downloads encyclopedia."
    logging.info("Downloading encyclopedia…")
    response = requests.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={
        "application_id": application_id,
        "fields": "tank_id,name",
    })
    obj = get_response_object(response)
    encyclopedia = [(tank["name"], tank["tank_id"]) for tank in obj["data"].values()]
    encyclopedia = sorted(encyclopedia, key=operator.itemgetter(1))
    return encyclopedia


def get_response_object(response):
    "Gets response object."
    response.raise_for_status()
    obj = response.json()
    if obj["status"] == "error":
        raise ValueError("{0[code]} {0[message]}".format(obj))
    return obj


def write_json(output, obj):
    "Writes serialized object to output."
    s = json.dumps(obj)
    output.write(LENGTH.pack(len(s)))
    output.write(s.encode("utf-8"))


if __name__ == "__main__":
    main()
