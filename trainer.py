#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse

import numpy
import scipy

import utils


def main(args):
    _, *tank_ids = next(args.stats)
    print("%d tanks." % len(tank_ids))
    print("%d features." % args.num_features)

    x_shape = (len(tank_ids), args.num_features)
    print("X shape: %r." % (x_shape, ))
    x = numpy.random.rand(*x_shape)
    theta = numpy.random.rand(*x_shape)

    alpha = 0.001

    for i, row in enumerate(args.stats, start=1):
        account_id, *row = row
        if i % 1000 == 0:
            print("Account #%d (account_id=%s)." % (i, account_id))


def do_step(x, theta, y, r):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="stats", help="input file", metavar="<stats.csv.gz>", type=utils.CsvReaderGZipFileType())
    parser.add_argument("--lambda", default=1.0, dest="lambda_", help="regularization parameter (default: %(default)s)", type=float)
    parser.add_argument("--num-features", default=16, dest="num_features", help="number of features (default: %(default)s)", type=int)
    main(parser.parse_args())
