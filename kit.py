#!/usr/bin/env python3
# coding: utf-8

import asyncio
import collections
import http.client
import itertools
import logging
import os
import sys

from datetime import datetime, timedelta
from functools import wraps
from operator import attrgetter, itemgetter
from time import time
from random import normalvariate

import aiohttp
import click


# Pre-defines.
# ------------------------------------------------------------------------------

MAX_IDS_PER_REQUEST = 100

AUTO_ADAPT_REQUEST_COUNT = 150
MAX_BUFFER_SIZE = 10000

MIN_PENDING_COUNT = 1
DEFAULT_PENDING_COUNT = 8
MAX_PENDING_COUNT = 32

TANK_ID_BLACKLIST = {64513, 64833, 64545}


# Entry point.
# ------------------------------------------------------------------------------

@click.group()
@click.option("-l", "--log-file", default=sys.stderr, help="Log file.", metavar="<file>", type=click.File("wt"))
def main(log_file):
    """Tankopoisk."""
    logging.basicConfig(
        format="%(asctime)s (%(module)s) %(levelname)s %(message)s",
        level=logging.INFO,
        stream=log_file,
        datefmt="%H:%M:%S",
    )


def run_in_event_loop(func):
    """Async command decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(asyncio.coroutine(func)(*args, **kwargs))
    return wrapper


# Commands.
# ------------------------------------------------------------------------------

MB = 1048576.0


@main.command()
@click.option("--app-id", default="demo", help="Application ID.", metavar="<application ID>", show_default=True)
@click.option("--start-id", default=1, help="Start account ID.", metavar="<account ID>", show_default=True, type=int)
@click.option("--end-id", default=40000000, help="End account ID.", metavar="<account ID>", show_default=True, type=int)
@click.argument("output", type=click.File("wb"))
@run_in_event_loop
def get(app_id: str, start_id: int, end_id: int, output):
    """Get account statistics dump."""
    api = Api(app_id)
    consumer = AccountTanksConsumer(start_id, output)
    max_pending_count = DEFAULT_PENDING_COUNT
    pending = set()
    start_time = time()
    # Main loop.
    for account_ids in chop(range(start_id, end_id + 1), MAX_IDS_PER_REQUEST):
        # Acquire buffer and schedule request.
        pending.add(asyncio.async(api.account_tanks(account_ids)))
        if len(pending) < max_pending_count:
            continue
        # Wait for the request completion.
        if len(consumer.buffer) < MAX_BUFFER_SIZE:
            done, pending = yield from asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        else:
            logging.warning("Maximum buffer size is reached.")
            done, pending = yield from asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
        # Process results.
        consumer.consume_all(done)
        # Adapt concurrent request count.
        max_pending_count = adapt_max_pending_count(api, max_pending_count)
        # Print runtime statistics.
        aps = (consumer.expected_id - start_id) / (time() - start_time)
        logging.info(
            "#%d (%d) buffer: %d | tanks: %d | aps: %.1f | apd: %.0f",
            consumer.expected_id, consumer.account_count, len(consumer.buffer), consumer.tank_count, aps, aps * 86400.0,
        )
    # Let the last pending tasks finish.
    logging.info("Finishing.")
    if pending:
        done, _ = yield from asyncio.wait(pending)
        consumer.consume_all(done)
    assert not consumer.buffer, "there are buffered results left"
    # Print total statistics.
    logging.info("Finished in %s.", timedelta(seconds=time() - start_time))
    logging.info("Dump size: %.1fMiB.", output.tell() / MB)
    logging.info("Last existing ID: %s.", consumer.last_existing_id)
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


@main.command()
@click.argument("input_", type=click.File("rb"))
def cat(input_):
    """Print dump contents."""
    while True:
        stats = read_account_stats(input_)
        if not stats:
            break  # end of file
        account_id, tanks = stats
        for tank in tanks:
            print(account_id, *tank)


@main.command()
@click.argument("old", type=click.File("rb"))
@click.argument("new", type=click.File("rb"))
@click.argument("output", type=click.File("wb"))
def diff(old, new, output):
    """Make difference dump of two dumps."""
    new.seek(0, os.SEEK_END)
    new_size = new.tell() / MB
    new.seek(0, os.SEEK_SET)

    old_stats, new_stats = enumerate_tanks(old), enumerate_tanks(new)
    diff_stats = enumerate_diff(old_stats, new_stats)

    account_count = tank_count = 0
    start_time = time()

    for i, (account_id, tanks) in enumerate(itertools.groupby(diff_stats, attrgetter("account_id"))):
        if i % 100 == 0:
            new_position = new.tell() / MB
            speed = new_position * 60.0 / (time() - start_time)
            logging.info(
                "#%d | old: %.1fMiB | new: %.1fMiB | acc: %d | tanks: %d | %.1f MiB/min | eta: %.1f min",
                i, old.tell() / MB, new_position, account_count, tank_count, speed, (new_size - new_position) / speed,
            )
        tank_count += write_account_stats(account_id, tanks, output)
        account_count += 1

    logging.info("Accounts: %d. Tanks: %d.", account_count, tank_count)


@main.command()
@click.option("--app-id", default="demo", help="Application ID.", metavar="<application ID>", show_default=True)
@click.argument("output", type=click.File("wt", encoding="utf-8"))
@run_in_event_loop
def renew(app_id, output):
    """Get encyclopedia.py."""
    api = Api(app_id)
    # Get tank list.
    logging.info("Getting tank list.")
    tanks = dict((yield from api.encyclopedia_tanks(fields="tank_id")))
    logging.info("%s tanks (with blacklisted).", len(tanks))
    # Delete blacklisted tanks.
    for tank_id in TANK_ID_BLACKLIST:
        del tanks[tank_id]
    logging.info("%s tanks.", len(tanks))
    # Get tank infos.
    logging.info("Getting tank infos.")
    fields = ",".join([
        "chassis_rotation_speed",
        "circular_vision_radius",
        "engine_power",
        "gun_damage_max",
        "gun_damage_min",
        "gun_piercing_power_max",
        "gun_piercing_power_min",
        "gun_rate",
        "is_gift",
        "is_premium",
        "level",
        "max_health",
        "name",
        "name_i18n",
        "nation",
        "radio_distance",
        "short_name_i18n",
        "speed_limit",
        "turret_armor_board",
        "turret_armor_fedd",
        "turret_armor_forehead",
        "turret_rotation_speed",
        "type",
        "vehicle_armor_board",
        "vehicle_armor_fedd",
        "vehicle_armor_forehead",
        "weight",
    ])
    for tank_ids in chop(tanks.keys(), MAX_IDS_PER_REQUEST):
        tankinfos = yield from api.encyclopedia_tankinfo(tank_ids, fields=fields)
        for tank_id, tankinfo in tankinfos:
            tanks[tank_id].update(tankinfo)
    # Patch strange names.
    tanks[3601].update({"short_name_i18n": "Pz.Jag. I", "name_i18n": "Panzerjager I"})
    tanks[4417]["short_name_i18n"] = "Renault G1R"
    tanks[15425]["short_name_i18n"] = "F72 AMX 30"
    tanks[15681]["short_name_i18n"] = "AMX 30 prototype"
    tanks[15937]["short_name_i18n"] = "RenaultR35"
    tanks[54289].update({"short_name_i18n": "Lowe", "name_i18n": "Lowe"})
    tanks[63297].update({"short_name_i18n": "F69 AMX13 57 100", "name_i18n": "F69 AMX13 57 100"})
    # Print encyclopedia.
    logging.info("Printing encyclopedia.")
    print("#!/usr/bin/env python", file=output)
    print("# coding: utf-8", file=output)
    print(file=output)
    print("\"\"\"", file=output)
    print("World of Tanks encyclopedia.", file=output)
    print("Autogenerated on %s by kit.py renew." % datetime.now().replace(microsecond=0), file=output)
    print("\"\"\"", file=output)
    print(file=output)
    output.write("TANKS = ")
    pretty_print(tanks, output)
    logging.info("Well done.")


# API helper.
# ------------------------------------------------------------------------------

class Api:
    """Wargaming Public API interface."""

    def __init__(self, app_id: str):
        self.app_id = app_id
        self.connector = aiohttp.TCPConnector()
        self.reset_error_rate()

    def reset_error_rate(self):
        self.request_count = self.request_limit_exceeded_count = 0

    @asyncio.coroutine
    def account_tanks(self, account_ids):
        """Gets account tanks."""
        data = yield from self.make_request(
            "account/tanks",
            account_id=self.make_comma_separated_list(account_ids),
            fields="statistics,tank_id",
        )
        # Return accounts tanks sorted by tank ID.
        return [
            (int(account_id), sorted(tanks, key=itemgetter("tank_id")) if tanks else None)
            for account_id, tanks in data.items()
        ]

    @asyncio.coroutine
    def encyclopedia_tanks(self, **kwargs):
        """
        Gets the tanks list.
        http://ru.wargaming.net/developers/api_reference/wot/encyclopedia/tanks/
        """
        data = yield from self.make_request("encyclopedia/tanks", **kwargs)
        return self.fix_encyclopedia_data(data)

    @asyncio.coroutine
    def encyclopedia_tankinfo(self, tank_ids, **kwargs):
        """
        Gets the tank information.
        http://ru.wargaming.net/developers/api_reference/wot/encyclopedia/tankinfo/
        """
        data = yield from self.make_request(
            "encyclopedia/tankinfo",
            tank_id=self.make_comma_separated_list(tank_ids),
            **kwargs
        )
        return self.fix_encyclopedia_data(data)

    @asyncio.coroutine
    def make_request(self, method: str, **kwargs):
        """Makes API request."""
        params = dict(kwargs, application_id=self.app_id)
        backoff = exponential_backoff(0.1, 600.0, 2.0, 0.1)
        for sleep_time in backoff:
            try:
                response = yield from asyncio.wait_for(aiohttp.request(
                    "GET",
                    "http://api.worldoftanks.ru/wot/%s/" % method,
                    params=params,
                    connector=self.connector,
                ), 10.0)
            except asyncio.TimeoutError:
                logging.warning("Timeout.")
                response = None
            except aiohttp.errors.ClientError:
                logging.warning("Client error.")
                response = None
            if response is None:
                pass  # do nothing
            elif response.status == http.client.OK:
                self.request_count += 1
                json = yield from response.json()
                if json["status"] == "ok":
                    return json["data"]
                if json["error"]["message"] == "REQUEST_LIMIT_EXCEEDED":
                    self.request_limit_exceeded_count += 1
                print(json)
                logging.warning("API error: %s", json["error"]["message"])
            else:
                logging.error("HTTP status: %d", response.status_code)
            logging.warning("sleep %.1fs", sleep_time)
            yield from asyncio.sleep(sleep_time)

    @staticmethod
    def make_comma_separated_list(items) -> str:
        return ",".join(map(str, items))

    @staticmethod
    def fix_encyclopedia_data(data: dict) -> dict:
        return [(int(tank_id), tank) for tank_id, tank in data.items()]


# Buffering.
# ------------------------------------------------------------------------------

class AccountTanksConsumer:
    """Consumes results of account/tanks API requests."""

    def __init__(self, start_id: int, output):
        self.expected_id = start_id
        self.output = output
        self.buffer = {}
        self.account_count = 0
        self.tank_count = 0
        self.last_existing_id = None

    def consume_all(self, tasks):
        for task in tasks:
            self.consume(task.result())

    def consume(self, result):
        """Consumes request result."""
        # Buffer account stats.
        for account_id, tanks in result:
            self.buffer[account_id] = tanks
        # Dump stored results.
        while self.expected_id in self.buffer:
            # Pop expected result.
            tanks = self.buffer.pop(self.expected_id)
            # Write account stats.
            if tanks:
                write_account_stats(self.expected_id, map(self.to_tank_instance, tanks), self.output)
                # Update stats.
                self.account_count += 1
                self.tank_count += len(tanks)
                self.last_existing_id = self.expected_id
            # Expect next account ID.
            self.expected_id += 1

    @staticmethod
    def to_tank_instance(tank: dict):
        """Makes Tank instance from JSON tank entry."""
        return Tank(tank["tank_id"], tank["statistics"]["battles"], tank["statistics"]["wins"])


# Helpers.
# ------------------------------------------------------------------------------


def exponential_backoff(minimum: float, maximum: float, factor: float, jitter: float):
    """Exponential Backoff Algorithm."""
    value = minimum
    while True:
        yield value
        value = value * factor + jitter * normalvariate(0.0, 1.0)
        if value > maximum:
            value = maximum
        elif value < minimum:
            value = minimum


def adapt_max_pending_count(api: Api, max_pending_count: int) -> int:
    """Adapt maximum pending request count basing on API error rate."""
    if api.request_limit_exceeded_count > max_pending_count:
        max_pending_count = max(max_pending_count - 1, MIN_PENDING_COUNT)
        logging.warning("Concurrent request count is decreased to: %d.", max_pending_count)
        api.reset_error_rate()
    elif api.request_count >= AUTO_ADAPT_REQUEST_COUNT:
        if api.request_limit_exceeded_count == 0:
            max_pending_count = min(max_pending_count + 1, MAX_PENDING_COUNT)
            logging.info("Concurrent request count is increased to: %d.", max_pending_count)
        api.reset_error_rate()
    return max_pending_count


def pretty_print(obj, fp, indent=""):
    """
    Replacement for pprint.
    """
    inner_indent = "    %s" % indent
    if isinstance(obj, dict):
        fp.write("{\n")
        for key, value in sorted(obj.items()):
            fp.write(inner_indent)
            pretty_print(key, fp, inner_indent)
            fp.write(": ")
            pretty_print(value, fp, inner_indent)
            fp.write(",\n")
        fp.write("%s}" % indent)
    elif isinstance(obj, str):
        fp.write("\"%s\"" % obj)
    else:
        fp.write("%r" % obj)


# Serialization.
# ------------------------------------------------------------------------------

def write_uvarint(value: int, fp):
    """Writes unsigned varint value."""
    assert value >= 0, value
    while True:
        value, byte = value >> 7, value & 0x7F
        if value:
            byte |= 0x80
        fp.write(bytes((byte, )))
        if not value:
            break


def read_uvarint(fp) -> int:
    """Reads unsigned varint value."""
    continue_, value, shift = True, 0, 0
    while continue_:
        byte = fp.read(1)[0]
        continue_, value, shift = byte & 0x80, value | ((byte & 0x7F) << shift), shift + 7
    return value


def read_uvarints(count, fp):
    """Reads several uvarints at once."""
    for _ in range(count):
        yield read_uvarint(fp)


def write_account_stats(account_id: int, tanks, fp) -> int:
    """Writes account stats into file."""
    tanks = list(tanks)
    fp.write(b">>")
    write_uvarint(account_id, fp)
    write_uvarint(len(tanks), fp)
    for tank in tanks:
        write_uvarint(tank.tank_id, fp)
        write_uvarint(tank.battles, fp)
        write_uvarint(tank.wins, fp)
    return len(tanks)


def read_account_stats(fp):
    """Reads account stats from file."""
    if not fp.read(2):
        return  # end of file
    account_id = read_uvarint(fp)
    tank_count = read_uvarint(fp)
    return account_id, [Tank(*read_uvarints(3, fp)) for _ in range(tank_count)]


# Enumeration.
# ------------------------------------------------------------------------------

Tank = collections.namedtuple("Tank", "tank_id battles wins")


class AccountTank(collections.namedtuple("AccountTank", "account_id tank_id battles wins")):

    def key(self):
        return (self.account_id, self.tank_id)


def chop(iterable, length: int):
    """Splits iterable into chunks."""
    iterable = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterable, length))
        if not chunk:
            break
        yield chunk


def safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def enumerate_tanks(fp):
    """Reads all tanks from file."""
    while True:
        stats = read_account_stats(fp)
        if not stats:
            return
        account_id, tanks = stats
        for tank in tanks:
            tank_id, battles, wins = tank
            yield AccountTank(account_id, tank_id, battles, wins)


def enumerate_diff(old_iterator, new_iterator):
    """Generates diff entries."""
    old_iterator, new_iterator = iter(old_iterator), iter(new_iterator)
    old, new = safe_next(old_iterator), safe_next(new_iterator)
    while old or new:
        if not new or (old and old.key() < new.key()):
            old = safe_next(old_iterator)
        elif not old or (new and new.key() < old.key()):
            yield new
            new = safe_next(new_iterator)
        else:
            if (
                # Work around strange API behaviors.
                new.battles > old.battles and
                new.wins >= old.wins and
                new.battles - old.battles >= new.wins - old.wins
            ):
                yield AccountTank(new.account_id, new.tank_id, new.battles - old.battles, new.wins - old.wins)
            old, new = safe_next(old_iterator), safe_next(new_iterator)


# Entry point.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
