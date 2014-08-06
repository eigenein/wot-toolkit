#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging
import pickle
import struct
import time

import collect
import trainer


def main(args):
    logging.info("Unpacking planes.")
    planes = pickle.load(args.planes)

    logging.info("Reading header.")
    header, (*magic, column_count, row_count, value_count) = read_header(args.input)
    logging.info("Rows: %d. Columns: %d. Values: %d.", row_count, column_count, value_count)
    if row_count != len(planes):
        logging.warning("Expected %d planes but found %d.", len(planes), row_count)

    logging.info("Initializing model.")
    model = trainer.Model(row_count, column_count, value_count, args.feature_count, args.lambda_)
    initialize_model(model, args.input, planes, args.accounts)
    model.prepare(0.5)

    logging.info("Initial RMSE: %.6f.", model.step(0.0))

    logging.info("Gradient descent.")
    gradient_descent(model)

    # logging.info("Writing output profile.")
    # write_output(args.output, row_count, column_count, args.feature_count, args.lambda_, final_cost, x, theta, mean)

    logging.info("Finished.")


def read_header(input):
    header = input.read(collect.HEADER_LENGTH)
    return header, struct.unpack(collect.HEADER_FORMAT, header)


def initialize_model(model, input, planes, accounts):
    try:
        for j in range(column_count):
            account_id, column = read_column(input)
            print(j, account_id, file=accounts)
            for plane_id, rating in column:
                i = planes[plane_id][0]
                y[i, j], z[i, j] = rating, True
            if j % 100000 == 0:
                logging.info("%d columns | %.1f%%", j, 100.0 * j / column_count)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    return model


def read_column(input):
    account_id, values = struct.unpack(collect.ROW_START_FORMAT, input.read(collect.ROW_START_LENGTH))
    return account_id, [struct.unpack(collect.RATING_FORMAT, input.read(collect.RATING_LENGTH)) for i in range(values)]


def gradient_descent(model):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("-i", "--input", help="input file", metavar="<my.wowpstats>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--lambda", default=0.0, dest="lambda_", help="regularization parameter (default: %(default)s)", metavar="<lambda>", type=float)
    parser.add_argument("--num-features", default=4, dest="feature_count", help="number of features (default: %(default)s)", metavar="<number of features>", type=int)
    parser.add_argument("-o", "--output", help="output profile", metavar="<my.wowpthetax>", required=True, type=argparse.FileType("wb"))
    parser.add_argument("--accounts", help="accounts list output", metavar="<accounts.txt>", required=True, type=argparse.FileType("wt"))
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)
    main(args)
