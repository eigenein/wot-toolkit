#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
import json
import logging
import pickle
import struct

import collect


def main(args):
    logging.info("Loading intervals.")
    intervals = json.load(args.intervals)

    header, (*magic, all_rows, columns, all_values) = read_header(args.input)
    logging.info("Rows: %d. Columns: %d. Values: %d.", all_rows, columns, all_values)
    args.output.write(header)

    values = rows = previous_values = 0
    current_interval = intervals.pop(0)
    for i in itertools.count():
        if i and (i % 10000 == 0):
            logging.info("#%d | values: %d", i, values)
        try:
            account_id, row, previous_values = repair_row(args.input, previous_values, current_interval)
        except ValueError as e:
            logging.fatal("At row #%d: %s.", i, e)
            break
        except StopIteration:
            break
        while account_id >= current_interval:
            current_interval = intervals.pop(0)
        collect.write_row(args.output, account_id, row)
        values += len(row)
        rows += 1

    if (rows, values) == (all_rows, all_values):
        logging.info("Successfully repaired.")
    else:
        logging.fatal("Failed to repair: %d values expected but %d read, %d rows expected but %d read.",
            all_values, values, all_rows, rows)


def read_header(input):
    header = input.read(collect.HEADER_LENGTH)
    return header, struct.unpack(collect.HEADER_FORMAT, header)


def repair_row(input, previous_values, current_interval):
    buffer = input.read(collect.ROW_START_LENGTH)
    if not buffer:
        raise StopIteration()
    account_id, values = struct.unpack(collect.ROW_START_FORMAT, buffer)
    if account_id <= 0:
        raise ValueError("invalid account ID: %d" % account_id)
    if account_id < current_interval:
        previous_values, values = values, values - previous_values
    else:
        previous_values = values
    if values <= 0:
        raise ValueError("invalid values: %d" % values)
    row = []
    for i in range(values):
        plane_id, rating = struct.unpack(collect.RATING_FORMAT, input.read(collect.RATING_LENGTH))
        if rating < 0.0 or rating > 1.0:
            raise ValueError("invalid rating: %f" % rating)
        row.append((plane_id, rating))
    return account_id, row, previous_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check stats file.")
    parser.add_argument("-i", "--input", help="input file", metavar="<my.wowpstats>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("-o", "--output", help="output file", metavar="<my.repaired.wowpstats>", required=True, type=argparse.FileType("wb"))
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--intervals", metavar="<intervals.json>", required=True, type=argparse.FileType("rt"))
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)
    main(args)
