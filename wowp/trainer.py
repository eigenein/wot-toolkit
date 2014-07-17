#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
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
    y, r = read_y(args.input, planes, rows, columns)
    logging.info("%d values read (%d expected).", r.sum(), values)

    logging.info("Feature normalization.")
    y, mean = normalize(y, r)

    logging.info("Initializing parameters.")
    x, theta = initialize_parameters(y.shape, args.num_features)

    logging.info("Gradient descent.")
    gradient_descent(y, r, x, theta, args.lambda_)


def read_y(input, planes, rows, columns):
    y = numpy.zeros((rows, columns), dtype=DTYPE)
    z = numpy.zeros((rows, columns), dtype=numpy.bool)
    try:
        for j in range(columns):
            column = read_column(input)
            for plane_id, rating in column:
                i = planes[plane_id][0]
                y[i, j], z[i, j] = rating, True
            if j % 100000 == 0:
                logging.info("%d columns | %.1f%%", j, 100.0 * j / columns)
    except KeyboardInterrupt:
        pass
    return y, z


def read_column(input):
    account_id, values = struct.unpack(collect.ROW_START_FORMAT, input.read(collect.ROW_START_LENGTH))
    return [struct.unpack(collect.RATING_FORMAT, input.read(collect.RATING_LENGTH)) for i in range(values)]


def normalize(y, r):
    mean = y.sum(1) / r.sum(1)
    mean = numpy.nan_to_num(mean)
    mean = mean.reshape((mean.size, 1))
    y -= mean
    y *= r
    return y, mean


def initialize_parameters(y_shape, num_features):
    x = numpy.random.rand(y_shape[0], num_features)
    x *= 0.001
    theta = numpy.random.rand(y_shape[1], num_features)
    theta *= 0.001
    return x, theta


def gradient_descent(y, r, x, theta, l):
    alpha = 0.001
    previous_cost = float("+inf")

    try:
        for i in itertools.count(1):
            logging.debug("Doing step.")
            x_new, theta_new = step(y, r, x, theta, l, alpha)
            logging.debug("Computing new cost.")
            new_cost = cost(y, r, x_new, theta_new, l)
            logging.info("#%d | alpha: %.3f | cost: %.3f | delta: %.6f", i, alpha, previous_cost, new_cost - previous_cost)
            if new_cost < previous_cost:
                x, theta = x_new, theta_new
                previous_cost = new_cost
                alpha *= 1.1
            else:
                alpha *= 0.5
    except KeyboardInterrupt:
        pass


def get_d(y, r, x, theta):
    return (x.dot(theta.T) - y) * r


def step(y, r, x, theta, l, alpha):
    d = get_d(y, r, x, theta)
    x_grad = d.dot(theta) + l * x
    d = d.T
    theta_grad = d.dot(x) + l * theta
    return (x - alpha * x_grad, theta - alpha * theta_grad)


def cost(y, r, x, theta, l):
    return (get_d(y, r, x, theta) ** 2).sum() / 2.0 + l * (theta ** 2).sum() / 2.0 + l * (x ** 2).sum() / 2.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("-i", "--input", help="input file", metavar="<my.wowpstats>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--lambda", default=1.0, dest="lambda_", help="regularization parameter (default: %(default)s)", metavar="<lambda>", type=float)
    parser.add_argument("--num-features", default=16, dest="num_features", help="number of features (default: %(default)s)", metavar="<number of features>", type=int)
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)
    main(args)
