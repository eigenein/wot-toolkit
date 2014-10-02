#!/usr/bin/env python3
# coding: utf-8

import collections
import difflib
import enum
import logging
import sys

import click

import wotstats


class Tag(enum.Enum):
    CHANGED = 0
    EQUAL = 1
    NEW = 2
    DROPPED = 3


@click.command(help="Compare two wotstats files.")
@click.argument("file1", metavar="<file-1.wotstats>", type=click.File("rb"))
@click.argument("file2", metavar="<file-2.wotstats>", type=click.File("rb"))
@click.option("-o", "--output", metavar="<diff.wotstats>", help="Output file.", required=True, type=click.File("wb"))
def main(file1, file2, output):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, stream=sys.stderr)

    logging.info("Reading files…")
    account_count1, item_count1, encyclopedia1, accounts1 = wotstats.read(file1)
    account_count2, item_count2, encyclopedia2, accounts2 = wotstats.read(file2)

    if encyclopedia1 != encyclopedia2:
        compare_encyclopedias(encyclopedia1, encyclopedia2)
        raise click.ClickException("encyclopedias differ")

    wotstats.write_header(output, 0, 0)
    wotstats.write_json(output, encyclopedia2)

    logging.info("File 1: %d accounts, %d items.", account_count1, item_count1)
    logging.info("File 2: %d accounts, %d items.", account_count2, item_count2)

    logging.info("Comparing files…")
    try:
        statistics, account_count, item_count = collections.Counter(), 0, 0
        for _, (account_id, items) in enumerate(compare_files(accounts1, accounts2, statistics), start=1):
            wotstats.write_account(output, account_id, items)
            account_count += 1
            item_count += len(items)
    finally:
        output.seek(0)
        wotstats.write_header(output, account_count, item_count)  # TODO: account_count, item_count
        logging.info("Accounts: %s. Items: %s.", account_count, item_count)
        log_statistics(statistics)

    logging.info("Finished.")


def compare_encyclopedias(encyclopedia1, encyclopedia2):
    "Compares two encyclopedias and prints diff."
    matcher = difflib.SequenceMatcher(a=encyclopedia1, b=encyclopedia2, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            print("-%d,%d +%d,%d" % (i1, i2, j1, j2))
        if tag in ("replace", "delete"):
            print("-", encyclopedia1[i1:i2])
        if tag in ("replace", "insert"):
            print("+", encyclopedia2[j1:j2])


def compare_files(accounts1, accounts2, statistics):
    "Compares two wotstats files."
    account1 = account2 = None
    while True:
        account1 = account1 or next_account(accounts1)
        account2 = account2 or next_account(accounts2)
        if account1 is None and account2 is None:
            break
        if account1 is None:
            logging.info("New account %d.", account2[0])
            statistics[Tag.NEW] += 1
            yield account2
            account2 = None  # Use account 2.
            continue
        if account2 is None:
            logging.warning("Drop account %d.", account1[0])
            statistics[Tag.DROPPED] += 1
            account1 = None  # Use (drop) account 1.
            continue
        # Compare two accounts.
        (account1_id, items1), (account2_id, items2) = account1, account2
        if account1_id < account2_id:
            logging.warning("Drop account %d.", account1_id)
            statistics[Tag.DROPPED] += 1
            account1 = None  # Use (drop) account 1.
            continue
        if account1_id > account2_id:
            logging.info("New account %d.", account2_id)
            statistics[Tag.NEW] += 1
            yield account2
            account2 = None  # Use account 2.
            continue
        account1 = account2 = None  # Use both account 1 and 2.
        # Two accounts with equal ID.
        account_id = account1_id
        if items1 == items2:
            statistics[Tag.EQUAL] += 1
            continue
        # Items differ.
        statistics[Tag.CHANGED] += 1
        try:
            items = list(compare_items(items1, items2))
        except ValueError as ex:
            logging.warning("Account #%s: %s.", account_id, str(ex))
        else:
            if items:
                yield account_id, items
            else:
                logging.warning("Account #%s: no items in diff.", account_id)


def compare_items(items1, items2):
    items1 = {item[0]: (item[1], item[2]) for item in items1}
    for row, battles2, wins2 in items2:
        if row in items1:
            battles1, wins1 = items1[row]
            if battles1 > battles2:
                logging.warning("row=%s, battles1=%s, battles2=%s", row, battles1, battles2)
                raise ValueError("battle count mismatch")
            if wins1 > wins2:
                logging.warning("row=%s, wins1=%s, wins2=%s", row, wins1, wins2)
                raise ValueError("win count mismatch")
            if battles1 != battles2:
                yield (row, battles2 - battles1, wins2 - wins1)
        else:
            yield (row, battles2, wins2)


def next_account(iterator):
    "Gets next account from the iterator."
    try:
        return next(iterator)
    except StopIteration:
        return None


def log_statistics(statistics):
    logging.info(
        "Changed: %d. Equal: %d. New: %d. Dropped: %d.",
        statistics[Tag.CHANGED], statistics[Tag.EQUAL], statistics[Tag.NEW], statistics[Tag.DROPPED],
    )


if __name__ == "__main__":
    main()
