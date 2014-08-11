#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
import logging
import operator
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

    initial_rmse, _, average_error, max_error = model.step(0.0)
    logging.info("Initial RMSE: %.6f.", initial_rmse)
    logging.info("Initial avg error: %.6f.", average_error)
    logging.info("Initial max error: %.6f.", max_error)

    logging.info("Gradient descent.")
    gradient_descent(model, initial_rmse)

    logging.info("Base: %.6f.", model.base)

    # Debug code for account #191824 (5589968).
    # ./train.py -i 20140729-2241.wowpstats --planes planes.pickle -o my.wowpthetax --accounts accounts.txt --num-features 256
    logging.info("Predicting.")
    predictions = [(plane[1], model.predict(plane[0], 191824)) for plane in planes.values()]
    predictions = sorted(predictions, key=operator.itemgetter(1), reverse=True)
    for name, prediction in predictions:
        logging.info("%020s: %4.1f", name, prediction)

    logging.info("Finished.")


def read_header(input):
    header = input.read(collect.HEADER_LENGTH)
    return header, struct.unpack(collect.HEADER_FORMAT, header)


def initialize_model(model, input, planes, accounts):
    index = 0
    try:
        for j in range(model.column_count):
            account_id, column = read_column(input)
            print(j, account_id, file=accounts)
            for plane_id, rating in column:
                i = planes[plane_id][0]
                model.set_value(index, i, j, rating)
                index += 1
            if j % 100000 == 0:
                logging.info("%d columns | %.1f%%", j, (100.0 * j) / model.column_count)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    return model


def read_column(input):
    account_id, values = struct.unpack(collect.ROW_START_FORMAT, input.read(collect.ROW_START_LENGTH))
    return account_id, [struct.unpack(collect.RATING_FORMAT, input.read(collect.RATING_LENGTH)) for i in range(values)]


def gradient_descent(model, initial_rmse):
    alpha, previous_rmse = 0.001, initial_rmse
    try:
        for iteration in itertools.count(1):
            model.shuffle()
            rmse, _, average_error, max_error = model.step(alpha)
            if alpha < 1e-09:
                logging.warning("Learning rate is too small. Stopping.")
                break
            logging.info(
                "#%d | a: %.9f | rmse %.6f | d_rmse: %.6f | avg: %.6f | max: %.6f",
                iteration, alpha, rmse, rmse - previous_rmse, average_error, max_error,
            )
            alpha *= 1.01 if rmse < previous_rmse else 0.5
            previous_rmse = rmse
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")


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
