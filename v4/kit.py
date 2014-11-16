#!/usr/bin/env python3
# coding: utf-8

import itertools
import logging
import sys

from operator import itemgetter
from time import sleep, time
from random import normalvariate

import click
import requests


@click.group()
def main():
    """Tankopoisk v4."""
    pass


@main.command()
@click.option("--app-id", default="demo", help="Application ID.", metavar="<application ID>", show_default=True)
@click.option("--start-id", default=1, help="Start account ID.", metavar="<account ID>", show_default=True, type=int)
@click.option("--end-id", default=40000000, help="End account ID.", metavar="<account ID>", show_default=True, type=int)
@click.option("--log-file", default=sys.stderr, help="Log file.", metavar="<file>", type=click.File("wt"))
@click.argument("output", type=click.File("wb"))
def get(app_id, start_id, end_id, log_file, output):
    """Get account statistics dump."""
    logging.basicConfig(format="%(asctime)s (%(module)s) %(levelname)s %(message)s", level=logging.INFO, stream=log_file)

    api, start_time = Api(app_id), time()
    for account_ids in chop(range(start_id, end_id), 100):
        account_tanks = sorted(api.account_tanks(account_ids), key=itemgetter(0))  # sort by account ID
        # Print statistics.
        aps = account_ids[-1] / (time() - start_time)
        logging.info("#%d aps: %.1f apd: %.0f", account_ids[-1], aps, aps * 86400.0)


class Api:
    """Wargaming Public API interface."""

    def __init__(self, app_id):
        self.app_id = app_id
        self.session = requests.Session()

    def account_tanks(self, account_ids):
        """Gets account tanks."""
        data = self.make_request(
            "account/tanks",
            account_id=self.make_account_id(account_ids),
            fields="statistics,tank_id",
        )
        for account_id, tanks in data.items():
            if tanks:
                # Sort by tank ID.
                yield int(account_id), sorted(tanks, key=itemgetter("tank_id"))

    def make_request(self, method, **kwargs):
        """Makes API request."""
        params = dict(kwargs, application_id=self.app_id)
        backoff = exponential_backoff(0.1, 600.0, 2.0, 0.1)
        for sleep_time in backoff:
            response = self.session.get("http://api.worldoftanks.ru/wot/%s/" % method, params=params)
            if response.status_code == requests.codes.ok:
                json = response.json()
                if json["status"] == "ok":
                    return json["data"]
                logging.warning("API error: %s", json["error"]["message"])
            else:
                logging.warning("HTTP status: %d", response.status_code)
            logging.warning("sleep %.1fs", sleep_time)
            sleep(sleep_time)


    @staticmethod
    def make_account_id(account_ids):
        """Makes account_id value."""
        return ",".join(map(str, account_ids))


def chop(iterable, length):
    """Splits iterable into chunks."""
    iterable = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterable, length))
        if not chunk:
            break
        yield chunk


def exponential_backoff(minimum, maximum, factor, jitter):
    """Exponential Backoff Algorithm."""
    value = minimum
    while True:
        yield value
        value = value * factor + jitter * normalvariate(0.0, 1.0)
        if value > maximum:
            value = maximum
        elif value < minimum:
            value = minimum


def write_uvarint(value, fp):
    """Writes unsigned varint value."""
    while True:
        value, byte = value >> 7, value & 0x7F
        if value:
            byte |= 0x80
        fp.write(bytes((byte, )))
        if not value:
            break


def read_uvarint(fp):
    """Reads unsigned varint value."""
    continue_, value, shift = True, 0, 0
    while continue_:
        byte = fp.read(1)[0]
        continue_, value, shift = byte & 0x80, value | ((byte & 0x7F) << shift), shift + 7
    return value


def write_account_stats(account_id, tanks, fp):
    """Writes account stats into file."""
    fp.write(b">>")
    write_uvarint(account_id, fp)
    write_uvarint(len(tanks), fp)
    for tank in tanks:
        write_uvarint(tank["tank_id"], fp)
        write_uvarint(tank["statistics"]["battles"], fp)
        write_uvarint(tank["statistics"]["wins"], fp)


if __name__ == "__main__":
    main()
