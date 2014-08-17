#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import concurrent.futures
import itertools
import logging
import operator
import random
import threading
import time

import click
import requests

import wotstats


@click.command(help="Download account database.")
@click.option("--application-id", default="demo", help="Application ID.", show_default=True)
@click.option("--threads", default=8, help="Thread count.", show_default=True, type=int)
@click.option("--min-battles", default=10, help="Minimum tank battles.", show_default=True, type=int)
@click.option("-o", "--output", help="Output file.", required=True, type=click.File("wb"))
@click.option("--log", default=sys.stderr, help="Log file.", type=click.File("wt"))
def main(application_id, threads, min_battles, output, log):
    # Initialize logging.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=log)
    # Write empty header.
    wotstats.write_header(output, 0, 0)
    # Download encyclopedia.
    encyclopedia = download_encyclopedia(application_id)
    wotstats.write_json(output, encyclopedia)
    # Download database.
    column_count, value_count = download_database(application_id, max(threads, 1), min_battles, encyclopedia, output)
    # Seek to the beginning and update header.
    output.seek(0)
    wotstats.write_header(output, column_count, value_count)
    # Finished.
    logging.info("Finished: %d columns, %d values.", column_count, value_count)


class Local(threading.local):
    "Provides thread-local session."
    def __init__(self):
        self.session = requests.Session()


def download_database(application_id, thread_count, min_battles, encyclopedia, output):
    "Downloads database."
    logging.info("Starting download…")
    # Reverse encyclopedia.
    reverse_encyclopedia = {tank[1]: row for row, tank in enumerate(encyclopedia)}
    # Initialize statistics.
    column_count = value_count = 0
    start_time = time.time()
    # Iterate over all accounts.
    executor, local = concurrent.futures.ThreadPoolExecutor(thread_count), Local()
    args = (",".join(map(str, range(i, i + 100))) for i in itertools.count(1, 100))
    try:
        while True:
            # Submit API requests.
            futures = [
                executor.submit(get_account_tanks, local, application_id, account_id)
                for account_id in itertools.islice(args, thread_count * 2)
            ]
            # Process results.
            for future in futures:
                result = future.result()
                # Iterate over accounts in result.
                items = sorted((int(account_id), tanks) for account_id, tanks in result["data"].items())
                for account_id, tanks in items:
                    account_id = int(account_id)
                    if tanks is None:
                        continue
                    tanks = [
                        (reverse_encyclopedia[tank["tank_id"]], tank["statistics"]["battles"], tank["statistics"]["wins"])
                        for tank in tanks if tank["statistics"]["battles"] >= min_battles
                    ]
                    if not tanks:
                        continue
                    value_count += wotstats.write_account(output, account_id, sorted(tanks))
                    column_count += 1
                # Print statistics.
                apd = 86400.0 * account_id / (time.time() - start_time)
                logging.info(
                    "#%d | %d acc. | apd: %.1f | %d val. | %.1fMiB",
                    account_id, column_count, apd, value_count, output.tell() / 1048576.0,
                )
    except KeyboardInterrupt:
        logging.warning("Interrupted.")
    except:
        logging.exception("Fatal error.")
    return column_count, value_count


def get_account_tanks(local, application_id, account_id):
    "Requests tanks for the specified account IDs."
    for attempt in range(10):
        if attempt:
            time.sleep(random.uniform(1.0, 5.0))  # sleep on next retries
        try:
            return get_response(local.session.get("http://api.worldoftanks.ru/wot/account/tanks/", timeout=10.0, params={
                "application_id": application_id,
                "account_id": account_id,
                "fields": "statistics,tank_id",
            }))
        except requests.exceptions.ConnectionError:
            logging.warning("Connection error.")
        except requests.exceptions.Timeout:
            logging.warning("Request timeout.")
        except ValueError as e:
            logging.warning("%s.", e)
        except KeyboardInterrupt:
            raise
        except:
            logging.exception("Can't get account tanks.")
    raise ValueError("all attempts failed")


def download_encyclopedia(application_id):
    "Downloads encyclopedia."
    logging.info("Downloading encyclopedia…")
    response = requests.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={
        "application_id": application_id,
        "fields": "tank_id,name",
    })
    obj = get_response(response)
    encyclopedia = [(tank["name"], tank["tank_id"]) for tank in obj["data"].values()]
    encyclopedia = sorted(encyclopedia, key=operator.itemgetter(1))
    return encyclopedia


def get_response(response):
    "Gets response object."
    response.raise_for_status()
    obj = response.json()
    if obj["status"] == "error":
        raise ValueError("{0[code]} {0[message]}".format(obj["error"]))
    return obj


if __name__ == "__main__":
    main()
