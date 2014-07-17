#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
import logging
import pickle
import struct
import time

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
    initial_cost = cost(y, r, x, theta, args.lambda_)[1]
    logging.info("Initial cost: %.6f.", initial_cost)

    logging.info("Gradient descent.")
    x, theta, final_cost = gradient_descent(y, r, x, theta, args.lambda_, initial_cost)
    logging.info("Cost improved by %.1fx.", initial_cost / final_cost)

    logging.info("Writing output profile.")
    write_output(args.output, rows, columns, args.num_features, args.lambda_, final_cost, x, theta, mean)

    logging.info("Finished.")


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
    x = numpy.random.rand(y_shape[0], num_features).astype(DTYPE)
    theta = numpy.random.rand(y_shape[1], num_features).astype(DTYPE)
    return x, theta


def gradient_descent(y, r, x, theta, l, initial_cost):
    alpha = 0.000001
    previous_cost = initial_cost
    values = r.sum()

    try:
        for i in itertools.count(1):
            start_time = time.time()
            logging.debug("Doing step.")
            x_new, theta_new = step(y, r, x, theta, l, alpha)
            logging.debug("Computing new cost.")
            d_sum, new_cost = cost(y, r, x_new, theta_new, l)
            logging.info("#%d | alpha: %.9f | cost: %.3f (%.6f) | avg. error: %.1f%% | %.1fs",
                i, alpha, new_cost, new_cost - previous_cost, 100.0 * d_sum / values, time.time() - start_time)
            if new_cost < previous_cost:
                x, theta = x_new, theta_new
                previous_cost = new_cost
                alpha *= 1.1
            else:
                alpha *= 0.5
    except KeyboardInterrupt:
        pass

    return x, theta, previous_cost


def get_d(y, r, x, theta):
    return (x.dot(theta.T) - y) * r


def step(y, r, x, theta, l, alpha):
    d = get_d(y, r, x, theta)
    x_grad = d.dot(theta) + l * x
    theta_grad = x.T.dot(d).T + l * theta
    return (x - alpha * x_grad, theta - alpha * theta_grad)


def cost(y, r, x, theta, l):
    d = get_d(y, r, x, theta).abs()
    d_sum = d.sum()
    d *= d  # elements squared
    return d_sum, d.sum() / 2.0 + l * (theta ** 2).sum() / 2.0 + l * (x ** 2).sum() / 2.0


def write_output(output, rows, columns, num_features, l, cost, x, theta, mean):
    output.write(b"wowpthetax")
    output.write(struct.pack("=hihff", rows, columns, num_features, l, cost))
    write_matrix(output, x)
    write_matrix(output, theta)
    write_matrix(output, mean)


def write_matrix(output, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output.write(struct.pack("=f", matrix[i, j]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("-i", "--input", help="input file", metavar="<my.wowpstats>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("--lambda", default=0.0, dest="lambda_", help="regularization parameter (default: %(default)s)", metavar="<lambda>", type=float)
    parser.add_argument("--num-features", default=4, dest="num_features", help="number of features (default: %(default)s)", metavar="<number of features>", type=int)
    parser.add_argument("-o", "--output", help="output profile", metavar="<my.wowpthetax>", type=argparse.FileType("wb"))
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)
    main(args)
