#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse

import numpy
import scipy

import utils


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="stats", help="input file", metavar="<stats.csv.gz>", type=utils.CsvReaderGZipFileType())
    parser.add_argument(default=1.0, dest="lambda_", help="regularization parameter (default: %(default)s)", type=float)
    parser.add_argument(default=16, dest="num_features", help="number of features (default: %(default)s)", type=int)
    main(parser.parse_args())
