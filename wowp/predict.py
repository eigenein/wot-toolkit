#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
import pickle
import struct

import numpy


def main(args):
    logging.info("Unpacking planes.")
    planes = pickle.load(args.planes)

    logging.info("Reading model.")
    x, theta, mean = read_model(args.input)

    logging.info("Predicting.")
    p = predict(x, theta, mean, args.account)
    items = sort(p, planes)

    print_items(items)


def read_model(input):
    magic = input.read(len(b"wowpthetax"))
    if magic == b"wowpthetax":
        logging.info("Magic is OK.")
    else:
        logging.warning("Invalid magic.")

    header = input.read(struct.calcsize("=hihff"))
    rows, columns, num_features, l, cost = struct.unpack("=hihff", header)

    x = read_matrix(input, rows, num_features)
    theta = read_matrix(input, columns, num_features)
    mean = read_matrix(input, rows, 1)

    return x, theta, mean


def read_matrix(input, rows, columns):
    matrix = numpy.zeros((rows, columns))
    size = struct.calcsize("=f")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = struct.unpack("=f", input.read(size))[0]
    return matrix


def predict(x, theta, mean, i):
    return x.dot(theta[i].T) + mean


def sort(p, planes):
    items = [(p[plane[0], 0], plane[1]) for plane_id, plane in planes.items()]
    return sorted(items, reverse=True)


def print_items(items):
    for rating, name in items:
        print("%5.1f%%  %s" % (rating * 100.0, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rating.")
    parser.add_argument("--planes", help="plane list", metavar="<planes.pickle>", required=True, type=argparse.FileType("rb"))
    parser.add_argument("-i", "--input", help="trained model", metavar="<my.wowpthetax>", required=True, type=argparse.FileType("rb"))
    parser.add_argument(dest="account", help="account sequence number", metavar="<number>", type=int)
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG)
    main(args)
