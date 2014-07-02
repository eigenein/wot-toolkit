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
    data = numpy.zeros(total_tank_number, dtype=DTYPE)
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
                data[tank_counter] = rating
                indices[tank_counter] = tank_rows[tank_id]
                tank_counter += 1
            # Log progress.
            position = args.input.tell()
            if i % 100000 == 0:
                logging.info("%d objects | %.1f MiB", i, position / 1048576.0)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    # Truncate arrays.
    logging.info("Tank counter: %d.", tank_counter)
    # Convert to matrix.
    y = scipy.sparse.csc_matrix((tank_number, account_number), dtype=DTYPE)
    y.data, y.indices, y.indptr = data, indices, indptr
    logging.info("Y: %r.", y)
    return y


def get_parameters(tank_number, account_number, feature_number):
    x = numpy.random.rand(tank_number, feature_number).astype(DTYPE)
    x *= 0.001
    logging.info("X shape: %r.", x.shape)
    theta = numpy.random.rand(account_number, feature_number).astype(DTYPE)
    theta *= 0.001
    logging.info("Theta shape: %r.", theta.shape)
    return x, theta


def gradient_descent(y, x, theta, l, num_iterations):
    logging.info("Gradient descent.")
    alpha, previous_cost = 0.001, float("+inf")
    try:
        for i in range(num_iterations):
            x_new, theta_new = step(x, theta, y, l, alpha)
            current_cost = cost(x_new, theta_new, y, l)
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


def cost(x, theta, y, l):
    diff, diff_t, x_theta = make_diff(x, theta, y)
    diff.data **= 2
    current_cost = diff.data.sum() / 2.0 + l * (theta ** 2).sum() / 2.0 + l * (x ** 2).sum() / 2.0
    diff.data **= 0.5
    revert(diff, x_theta)
    return current_cost


def step(x, theta, y, l, alpha):
    diff, diff_t, x_theta = make_diff(x, theta, y)

    x_grad = diff.dot(theta) + l * x
    theta_grad = diff_t.dot(x) + l * theta

    x_new = x - alpha * x_grad
    theta_new = theta - alpha * theta_grad

    revert(diff, x_theta)
    return (x_new, theta_new)


def nonzero(y):
    for col in range(y.shape[1]):
        for i in range(y.indptr[col], y.indptr[col + 1]):
            yield i, y.indices[i], col


def make_diff(x, theta, y):
    # Make transposed sparse matrix.
    y_t = scipy.sparse.csr_matrix(y.shape[::-1], dtype=DTYPE)
    y_t.data, y_t.indices, y_t.indptr = y.data, y.indices, y.indptr
    # Make diff.
    y.data *= -1  # diff = -y
    x_theta = numpy.ndarray(y.data.size, dtype=DTYPE)
    for i, row, col in nonzero(y):  # TODO: optimize
        x_theta[i] = (x[row] * theta[col]).sum()  # TODO: optimize
    y.data += x_theta  # diff = x.dot(theta.T) * r - y
    # Return both diff and transposed diff.
    return y, y_t, x_theta


def revert(diff, x_theta):
    # Revert make_diff.
    diff.data -= x_theta
    diff.data *= -1


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
