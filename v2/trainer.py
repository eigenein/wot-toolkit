#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import json
import logging

import msgpack
import requests
import numpy
import scipy.sparse


DTYPE = numpy.float32


def main(args):
    tanks = get_tanks(args.tanks)
    logging.info("Tanks: %d.", len(tanks))

    logging.info("Make rating matrix.")
    y = get_rating_matrix(args.input, tanks, args.account_number, args.total_tank_number)
    x, theta = get_parameters(len(tanks), args.account_number, args.feature_number)
    gradient_descent(y, x, theta, args.l, args.iteration_number)


def get_tanks(tanks):
    tanks = json.load(tanks)
    tanks = {int(tank_id): (tank["row"], tank["name"]) for tank_id, tank in tanks.items()}
    return tanks


def get_rating_matrix(input, tanks, account_number, total_tank_number):
    tank_number = len(tanks)
    # Initialize arrays.
    data = numpy.zeros(total_tank_number, dtype=DTYPE)
    indices = numpy.zeros(total_tank_number, numpy.int)
    indptr = numpy.zeros(account_number + 1, numpy.int)
    # Fill up arrays.
    tank_counter = 0
    try:
        for i, obj in enumerate(msgpack.Unpacker(args.input)):
            if i == account_number:
                logging.warning("Not all objects are read.")
                break
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
                indices[tank_counter] = tanks[tank_id][0]
                tank_counter += 1
            # Log progress.
            position = args.input.tell()
            if i % 10000 == 0:
                logging.info("Reading: %d objects | %.1f MiB", i, position / 1048576.0)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    indptr[account_number] = tank_counter
    # Truncate arrays.
    logging.info("Tank counter: %d.", tank_counter)
    # Convert to matrix.
    y = scipy.sparse.csc_matrix((tank_number, account_number), dtype=DTYPE)
    y.data, y.indices, y.indptr = data, indices, indptr
    logging.info("Y: %r.", y)
    return y


def get_parameters(tank_number, account_number, feature_number):
    logging.info("Generate initial parameters.")
    x = numpy.random.rand(tank_number, feature_number).astype(DTYPE)
    x *= 0.001
    logging.info("X shape: %r.", x.shape)
    theta = numpy.random.rand(account_number, feature_number).astype(DTYPE)
    theta *= 0.001
    logging.info("Theta shape: %r.", theta.shape)
    return x, theta


def gradient_descent(y, x, theta, l, iteration_number):
    logging.info("Gradient descent.")
    alpha, previous_cost = 1.0, float("+inf")
    try:
        for i in range(iteration_number):
            logging.info("Starting iteration #%d.", i)
            current_cost = 0.0

            logging.info("Generating column order.")
            columns = numpy.random.permutation(y.shape[1])

            logging.info("Copying x and theta.")
            x_new, theta_new = x.copy(), theta.copy()

            for j, column in enumerate(columns):
                # Get partial matrices.
                y_partial = getcol(y, column)
                r_partial = (y_partial != 0)
                # Compute partial cost.
                current_cost += cost(x_new, theta[column], y_partial, r_partial, l)
                # Compute partial x and theta.
                x_new, theta_new[column] = step(x_new, theta[column], y_partial, r_partial, l, alpha)
                # Print current iteration info and check current cost.
                if j % 1000 == 0:
                    logging.info("#%d/%d | cost: %.3f | prev: %.3f", i, j, current_cost, previous_cost)
                    if current_cost > previous_cost:
                        logging.warning("Current cost is greater than previous cost. Break.")
                        break

            logging.info(
                "#%d | cost: %.3f | delta: %.6f | alpha: %f",
                i, current_cost, current_cost - previous_cost, alpha)
            if current_cost < previous_cost:
                alpha *= 1.05
                logging.info("Alpha is increased up to %.6f.", alpha)
                x, theta = x_new, theta_new
                previous_cost = current_cost
            else:
                logging.warning("Reset alpha.")
                alpha *= 0.5
    except KeyboardInterrupt:
        logging.warning("Gradient descent is interrupted by user.")


def getcol(y, column):
    # Work around ineffective csc_matrix.getcol.
    data = y.data[y.indptr[column]:y.indptr[column + 1]]
    indices = y.indices[y.indptr[column]:y.indptr[column + 1]]
    return scipy.sparse.csr_matrix((data, indices, numpy.array([0, data.size])), shape=(1, y.shape[0])).toarray().T


def cost(x, theta, y, r, l):
    theta = theta.reshape((1, theta.size))
    return (((x.dot(theta.T) - y) * r) ** 2).sum() / 2.0 + l * (theta ** 2).sum() / 2.0 + l * (x ** 2).sum() / 2.0


def step(x, theta, y, r, l, alpha):
    theta = theta.reshape((1, theta.size))
    diff = (x.dot(theta.T) - y) * r
    x_grad = diff.dot(theta) + l * x
    theta_grad = diff.T.dot(x) + l * theta
    return (x - alpha * x_grad, theta - alpha * theta_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer.")
    parser.add_argument(
        dest="input",
        help="input file",
        metavar="<input.msgpack>",
        type=argparse.FileType("rb"),
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
        dest="feature_number",
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
        dest="iteration_number",
        help="number of iterations",
        metavar="<number>",
        type=int,
    )
    parser.add_argument(
        "--tanks",
        dest="tanks",
        help="tank list (from API)",
        metavar="<tanks.json>",
        required=True,
        type=argparse.FileType("rt"),
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
