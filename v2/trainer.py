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
    y_t = get_rating_matrix(args.input, tanks, args.account_number, args.total_tank_number)
    x_t, theta_t = get_parameters(len(tanks), args.account_number, args.feature_number)
    gradient_descent(y_t, x_t, theta_t, args.l, args.iteration_number, args.batch_size)


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
            # Append row.
            indptr[i] = tank_counter
            # Append row values.
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
    # Finish the last row.
    indptr[account_number] = tank_counter
    # Truncate arrays.
    logging.info("Tank counter: %d.", tank_counter)
    # Convert to matrix.
    y_t = scipy.sparse.csr_matrix((account_number, tank_number), dtype=DTYPE)
    y_t.data, y_t.indices, y_t.indptr = data, indices, indptr
    logging.info("Y': %r.", y_t)
    return y_t


def get_parameters(tank_number, account_number, feature_number):
    logging.info("Generate initial parameters.")
    x_t = numpy.random.rand(feature_number, tank_number).astype(DTYPE)
    x_t *= 0.001
    logging.info("X' shape: %r.", x_t.shape)
    theta_t = numpy.random.rand(feature_number, account_number).astype(DTYPE)
    theta_t *= 0.001
    logging.info("Theta' shape: %r.", theta_t.shape)
    return x_t, theta_t


def gradient_descent(y_t, x_t, theta_t, l, iteration_number, batch_size, step_size=100):
    logging.info("Gradient descent.")
    alpha = 0.001
    current_cost, previous_cost = 0.0, float("+inf")
    try:
        for i in range(0, iteration_number, step_size):
            logging.info("Starting step %d.", i)
            x_t_new, theta_t_new, current_cost = x_t.copy(), theta_t.copy(), 0.0
            for j in range(step_size):
                user = numpy.random.choice(y_t.shape[0] - batch_size + 1)
                users = slice(user, user + batch_size)
                # Get partial matrices.
                y_t_users = y_t[users, :].toarray()
                r_t_users = (y_t_users != 0)
                # Compute partial cost.
                current_cost += cost(x_t_new, theta_t_new[:, users], y_t_users, r_t_users, l)
                # Compute partial x and theta.
                x_t_new, theta_t_new[:, users] = step(x_t_new, theta_t_new[:, users], y_t_users, r_t_users, l, alpha)
            # Check current cost.
            logging.info("#%d alpha: %.6f | cost: %.6f | prev: %.6f | delta: %.6f", i, alpha, current_cost, previous_cost, current_cost - previous_cost)
            if current_cost < previous_cost:
                alpha *= 1.05
                previous_cost = current_cost
                x_t, theta_t = x_t_new, theta_t_new
            else:
                alpha *= 0.95
            del x_t_new
            del theta_t_new
    except KeyboardInterrupt:
        logging.warning("Gradient descent is interrupted by user.")


def cost(x_t, theta_t, y_t, r_t, l):
    diff_t = (theta_t.T.dot(x_t) - y_t) * r_t
    return (diff_t ** 2).sum() / 2.0 + l * (theta_t ** 2).sum() / 2.0 + l * (x_t ** 2).sum() / 2.0


def step(x_t, theta_t, y_t, r_t, l, alpha):
    diff_t = (theta_t.T.dot(x_t) - y_t) * r_t
    x_t_grad = theta_t.dot(diff_t) + l * x_t
    theta_t_grad = x_t.dot(diff_t.T) + l * theta_t
    return (x_t - alpha * x_t_grad, theta_t - alpha * theta_t_grad)


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
        default=10000,
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
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1000,
        dest="batch_size",
        help="gradient descent batch size (default: %(default)s)",
        metavar="<number of users>",
        type=int,
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
