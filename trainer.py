#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import json

import numpy
import scipy.sparse

import utils


def main(args):
    _, *tank_ids = next(args.stats)
    num_tanks = len(tank_ids)
    print("[ OK ] %d tanks." % num_tanks)
    print("[INFO] %d features." % args.num_features)

    print("[INFO] Reading stats.")
    y_shape = (num_tanks, args.num_accounts)
    print("[INFO] Y shape: %r." % (y_shape, ))
    account_ids, y, r = [], numpy.ndarray(y_shape), numpy.ndarray(y_shape)  # TODO: numpy.zeros

    for j, row in enumerate(args.stats):
        if j % 10000 == 0:
            print("[ OK ] %d rows read." % j)
        if j == args.num_accounts:
            break

        account_id, *row = row
        account_ids.append(int(account_id))

        for i, rating in enumerate(row):
            if rating:
                y[i, j] = float(rating)
                r[i, j] = 1.0

    print("[INFO] Feature normalization.")
    mean = y.sum(1) / r.sum(1)
    mean = numpy.nan_to_num(mean)
    mean = mean.reshape((mean.size, 1))
    y = (y - mean) * r

    x = 0.001 * numpy.random.rand(num_tanks, args.num_features)
    print("[INFO] X shape: %r." % (x.shape, ))
    theta = 0.001 * numpy.random.rand(args.num_accounts, args.num_features)
    print("[INFO] Theta shape: %r." % (theta.shape, ))

    alpha, previous_cost = 0.001, float("+inf")

    print("[INFO] Gradient descent.")
    try:
        for i in range(args.num_iterations):
            x_new, theta_new = do_step(x, theta, y, r, args.lambda_, alpha)
            current_cost = cost(x_new, theta_new, y, r, args.lambda_)

            print("[INFO] Step #%d." % i)
            print("[INFO] Cost: %.3f (%.3f)." % (current_cost, previous_cost))
            print("[INFO] Alpha: %f." % alpha)

            if current_cost < previous_cost:
                alpha *= 1.05
                x, theta = x_new, theta_new
            else:
                print("[WARN] Step: #%d." % i)
                print("[WARN] Reset alpha: %f." % alpha)
                print("[WARN] Cost: %.3f." % current_cost)
                alpha *= 0.5

            previous_cost = current_cost
    except KeyboardInterrupt:
        print("[WARN] Gradient descent is interrupted by user.")

    print("[INFO] Writing profile.")
    json.dump({"x": x.tolist(), "theta": theta.tolist(), "mean": mean.tolist(), "tanks": tank_ids}, args.profile, indent=2)
    print("[ OK ] Written profile.")


def cost(x, theta, y, r, lambda_):
    return (((x.dot(theta.T) - y) * r) ** 2).sum() / 2.0 + lambda_ * (theta ** 2).sum() / 2.0 + lambda_ * (x ** 2).sum() / 2.0


def do_step(x, theta, y, r, lambda_, alpha):
    diff = (x.dot(theta.T) - y) * r
    x_grad = diff.dot(theta) + lambda_ * x
    theta_grad = diff.T.dot(x) + lambda_ * theta
    return (x - alpha * x_grad, theta - alpha * theta_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="stats", help="input file", metavar="<stats.csv.gz>", type=utils.CsvReaderGZipFileType())
    parser.add_argument("--profile", dest="profile", help="output profile", metavar="<profile.json>", required=True, type=argparse.FileType("wt"))
    parser.add_argument("--lambda", default=1.0, dest="lambda_", help="regularization parameter (default: %(default)s)", metavar="<lambda>", type=float)
    parser.add_argument("--num-features", default=16, dest="num_features", help="number of features (default: %(default)s)", metavar="<number of features>", type=int)
    parser.add_argument("--num-accounts", default=500000, dest="num_accounts", help="number of accounts to read (default: %(default)s)", metavar="<number of accounts>", type=int)
    parser.add_argument("--num-iterations", default=100, dest="num_iterations", help="number of gradient descent iterations (default: %(default)s)", metavar="<number of iterations>", type=int)
    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        pass
