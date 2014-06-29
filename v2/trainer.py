#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import logging

import msgpack
import requests
import numpy
import scipy.sparse

import shared


DTYPE = numpy.float32


def main(args):
    tank_rows = get_tank_rows()
    logging.info("Tank rows: %d entries.", len(tank_rows))

    logging.info("Make rating matrix.")
    y = get_rating_matrix(args.input, tank_rows, args.account_number, args.total_tank_number)
    x, theta = get_parameters(len(tank_rows), args.account_number, args.num_features)
    gradient_descent(y, x, theta, args.l, args.num_iterations)


def get_tank_rows():
    response = requests.get(
        "http://api.worldoftanks.ru/wot/encyclopedia/tanks/",
        params={
            "application_id": shared.APPLICATION_ID,
            "fields": "tank_id",
        },
    )
    response.raise_for_status()
    tank_ids = sorted(map(int, response.json()["data"]))
    return {tank_id: i for i, tank_id in enumerate(tank_ids)}


def get_rating_matrix(input, tank_rows, account_number, total_tank_number):
    tank_number = len(tank_rows)
    # Initialize arrays.
    y_data = numpy.zeros(total_tank_number, dtype=DTYPE)
    indices = numpy.zeros(total_tank_number, numpy.int)
    indptr = numpy.zeros(account_number + 1, numpy.int)
    # Fill up arrays.
    tank_counter = 0
    try:
        for i, obj in enumerate(msgpack.Unpacker(args.input)):
            # Unpack object.
            account_id, *stats = obj
            # Append column.
            indptr[i] = tank_counter
            # Append column values.
            for j in range(0, len(stats), 3):
                tank_id, wins, delta = stats[j:j+3]
                battles = 2 * wins - delta
                rating = wins / battles
                # Append values.
                y_data[tank_counter] = rating
                indices[tank_counter] = tank_rows[tank_id]
                tank_counter += 1
            # Log progress.
            position = args.input.tell()
            if i % 100000 == 0:
                logging.info("%d objects | %.1f MiB", i, position / 1048576.0)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    indptr[account_number] = tank_counter
    # Truncate arrays.
    logging.info("Tank counter: %d.", tank_counter)
    # Convert to matrices.
    y = scipy.sparse.csc_matrix((tank_number, account_number), dtype=DTYPE)
    y.data, y.indices, y.indptr = y_data, indices, indptr
    logging.info("Y: %r.", y)
    return y


def get_parameters(tank_number, account_number, feature_number):
    x = 0.001 * numpy.random.rand(tank_number, feature_number)
    logging.info("X shape: %r.", x.shape)
    theta = 0.001 * numpy.random.rand(account_number, feature_number)
    logging.info("Theta shape: %r.", theta.shape)
    return x, theta


def gradient_descent(y, x, theta, l, num_iterations):
    logging.info("Gradient descent.")
    alpha, previous_cost = 0.001, float("+inf")
    try:
        for i in range(num_iterations):
            x_new, theta_new = step(y, x, theta, l, alpha)
            current_cost = cost(y, x_new, theta_new, l)
            logging.info(
                "#%d | cost: %.3f | delta: %.6f | alpha: %f",
                i, current_cost, current_cost - previous_cost, alpha)
            if current_cost < previous_cost:
                alpha *= 1.05
                x, theta = x_new, theta_new
            else:
                logging.warning("Step: #%d.", i)
                logging.warning("Reset alpha: %f.", alpha)
                logging.warning("Cost: %.3f.", current_cost)
                alpha *= 0.5
            previous_cost = current_cost
    except KeyboardInterrupt:
        logging.warning("Gradient descent is interrupted by user.")


def cost(y, x, theta, l):
    diff_data = get_diff(y, x, theta).data
    diff_data *= diff_data  # in-place sqr
    return diff_data.sum() / 2.0 + l * (theta ** 2).sum() / 2.0 + l * (x ** 2).sum() / 2.0


def step(y, x, theta, l, alpha):
    diff = get_diff(y, x, theta)
    x_grad = diff.dot(theta) + l * x
    diff.row, diff.col = diff.col, diff.row  # in-place transpose
    theta_grad = diff.dot(x) + l * theta
    return (x - alpha * x_grad, theta - alpha * theta_grad)


def get_diff(y, x, theta):
    diff = y.tocoo()  # copy
    diff *= -1.0  # -y
    diff.data += numpy.sum(x[diff.row] * theta[diff.col], 1)  # x.dot(theta.T) * r
    return diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer.")
    parser.add_argument(
        dest="input",
        help="input file",
        metavar="<input.msgpack.gz>",
        type=shared.GZipFileType("rb"),
    )
    parser.add_argument(
        "-a",
        dest="account_number",
        help="exact account number",
        metavar="<number>",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-t",
        dest="total_tank_number",
        help="exact total tank number (across all accounts)",
        metavar="<number>",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num-features",
        default=4,
        dest="num_features",
        help="features number (default: %(default)s)",
        metavar="<number>",
        type=int,
    )
    parser.add_argument(
        "-l",
        "--lambda",
        default=1.0,
        dest="l",
        help="regularization parameter (default: %(default)s)",
        metavar="<lambda>",
        type=float,
    )
    parser.add_argument(
        "--num-iterations",
        default=1000,
        dest="num_iterations",
        help="number of iterations",
        metavar="<number>",
        type=int,
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
