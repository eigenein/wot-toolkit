#!/usr/bin/env python3
# coding: utf-8

import asyncio
import http.client
import itertools
import logging
import sys

from datetime import timedelta
from functools import wraps
from operator import itemgetter
from time import time
from random import normalvariate

import aiohttp
import click


MAX_ACCOUNTS_PER_REQUEST = 100
MAX_PENDING_COUNT = 20


@click.group()
@click.option("--log-file", default=sys.stderr, help="Log file.", metavar="<file>", type=click.File("wt"))
def main(log_file):
    """Tankopoisk v4."""
    logging.basicConfig(format="%(asctime)s (%(module)s) %(levelname)s %(message)s", level=logging.INFO, stream=log_file)


def run_in_event_loop(func):
    """Async command decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(asyncio.coroutine(func)(*args, **kwargs))
    return wrapper


@main.command()
@click.option("--app-id", default="demo", help="Application ID.", metavar="<application ID>", show_default=True)
@click.option("--start-id", default=1, help="Start account ID.", metavar="<account ID>", show_default=True, type=int)
@click.option("--end-id", default=40000000, help="End account ID.", metavar="<account ID>", show_default=True, type=int)
@click.argument("output", type=click.File("wb"))
@run_in_event_loop
def get(app_id, start_id, end_id, output):
    """Get account statistics dump."""
    api = Api(app_id)
    consumer = AccountTanksConsumer(start_id, output, MAX_PENDING_COUNT * MAX_ACCOUNTS_PER_REQUEST)
    pending = set()
    start_time = time()
    # Main loop.
    for account_ids in chop(range(start_id, end_id + 1), MAX_ACCOUNTS_PER_REQUEST):
        pending.add(asyncio.async(api.account_tanks(account_ids)))
        if len(pending) < 4:
            continue
        done, pending = yield from asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        yield from consumer.consume_all(done)
        # Print runtime statistics.
        aps = (consumer.expected_id - start_id) / (time() - start_time)
        logging.info(
            "#%d (%d) buffered: %d | tanks: %d | aps: %.1f | apd: %.0f",
            consumer.expected_id, consumer.account_count, len(consumer.buffer), consumer.tank_count, aps, aps * 86400.0,
        )
    # Let the last pending tasks finish.
    logging.info("Finishing.")
    while pending:
        done, pending = yield from asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        yield from consumer.consume_all(done)
    assert not consumer.buffer, "there are buffered results left"
    # Print total statistics.
    logging.info("Finished in %s.", timedelta(seconds=time() - start_time))
    logging.info("Dump size: %.1fMiB.", output.tell() / 1048576.0)
    if not consumer.account_count:
        return
    logging.info(
        "Accounts: %d. Tanks: %d. Tanks per account: %.1f.",
        consumer.account_count, consumer.tank_count, consumer.tank_count / consumer.account_count,
    )
    logging.info(
        "%.0fB per account. %.1fB per tank.",
        output.tell() / consumer.account_count, output.tell() / consumer.tank_count,
    )


class Api:
    """Wargaming Public API interface."""

    def __init__(self, app_id):
        self.app_id = app_id
        self.connector = aiohttp.TCPConnector()

    @asyncio.coroutine
    def account_tanks(self, account_ids):
        """Gets account tanks."""
        data = yield from self.make_request(
            "account/tanks",
            account_id=self.make_account_id(account_ids),
            fields="statistics,tank_id",
        )
        # Return accounts tanks sorted by tank ID.
        return [
            (int(account_id), sorted(tanks, key=itemgetter("tank_id")) if tanks else None)
            for account_id, tanks in data.items()
        ]

    @asyncio.coroutine
    def make_request(self, method, **kwargs):
        """Makes API request."""
        params = dict(kwargs, application_id=self.app_id)
        backoff = exponential_backoff(0.1, 600.0, 2.0, 0.1)
        for sleep_time in backoff:
            response = yield from asyncio.wait_for(aiohttp.request(
                "GET",
                "http://api.worldoftanks.ru/wot/%s/" % method,
                params=params,
                connector=self.connector,
            ), 10.0)
            if response.status == http.client.OK:
                json = yield from response.json()
                if json["status"] == "ok":
                    return json["data"]
                logging.warning("API error: %s", json["error"]["message"])
            else:
                logging.warning("HTTP status: %d", response.status_code)
            logging.warning("sleep %.1fs", sleep_time)
            asyncio.sleep(sleep_time)

    @staticmethod
    def make_account_id(account_ids):
        """Makes account_id value."""
        return ",".join(map(str, account_ids))


class AccountTanksConsumer:
    """Consumes results of account/tanks API requests."""

    def __init__(self, start_id, output, buffer_size):
        self.expected_id = start_id
        self.output = output
        self.semaphore = asyncio.Semaphore(buffer_size)
        self.buffer = {}
        self.account_count = 0
        self.tank_count = 0

    @asyncio.coroutine
    def consume_all(self, tasks):
        for task in tasks:
            yield from self.consume(task.result())

    @asyncio.coroutine
    def consume(self, result):
        """Consumes request result."""

        # Sort by account ID.
        account_tanks = sorted(result, key=itemgetter(0))
        # Iterate through account stats.
        for account_id, tanks in account_tanks:
            yield from self.semaphore.acquire()
            self.buffer[account_id] = tanks
            # Dump stored results.
            while self.expected_id in self.buffer:
                # Pop expected result.
                tanks = self.buffer.pop(self.expected_id)
                self.semaphore.release()
                if tanks:
                    write_account_stats(account_id, tanks, self.output)
                    # Update stats.
                    self.account_count += 1
                    self.tank_count += len(tanks)
                # Expect next account ID.
                self.expected_id += 1


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
