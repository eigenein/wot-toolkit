#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging
import pickle
import struct

import numpy

import collect
import repair


DTYPE = numpy.float32


def main(args):
    logging.info("Unpacking planes.")
    planes = pickle.load(args.planes)

    logging.info("Reading header.")
    header, (*magic, columns, rows, values) = repair.read_header(args.input)
    logging.info("Rows: %d. Columns: %d. Values: %d.", rows, columns, values)
    if rows != len(planes):
        logging.warning("Expected %d planes but found %d.", len(planes), rows)

    logging.info("Reading rating matrix.")
    y, z = read_y(args.input, planes, rows, columns)
    logging.info("%d values read (%d expected).", z.sum(), values)


def read_y(input, planes, rows, columns):
    y = numpy.zeros((rows, columns), dtype=DTYPE)
    z = numpy.zeros((rows, columns), dtype=numpy.bool)
    for j in range(columns):
        column = read_column(input)
        for plane_id, rating in column:
            i = planes[plane_id][0]
            y[i, j] = rating
            z[i, j] = True
        if j % 100000 == 0:
            logging.info("%d columns | %.1f%%", j, 100.0 * j / columns)
    return y, z


def read_column(input):
    account_id, values = struct.unpack(collect.ROW_START_FORMAT, input.read(collect.ROW_START_LENGTH))
    return [struct.unpack(collect.RATING_FORMAT, input.read(collect.RATING_LENGTH)) for i in range(values)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("-i", "--input", help="input file", metavar="<my.wowpstats>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)
    main(args)
