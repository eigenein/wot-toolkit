#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import json

import numpy
import requests


def main(args):
    print("[INFO] Loading profile.")
    profile = json.load(args.profile)
    x = numpy.array(profile["x"])
    mean = numpy.array(profile["mean"])

    print("[INFO] Get tanks.")
    response = requests.get("http://api.worldoftanks.ru/wot/encyclopedia/tanks/", params={"application_id": "demo"})
    response.raise_for_status()
    data = response.json()["data"]
    tank_ids = {value["name"]: tank_id for tank_id, value in data.items()}
    tank_names = {tank_id: value["name"] for tank_id, value in data.items()}

    print("[INFO] Loading user stats.")
    stats = json.load(args.stats)
    y = numpy.zeros((x.shape[0], 1))
    r = numpy.zeros(y.shape)
    for i in range(y.shape[0]):
        tank_id = profile["tanks"][i]
        tank_name = tank_names[tank_id]
        if tank_name in stats:
            rating = stats[tank_name]
            y[i][0] = rating - mean[i][0]
            r[i][0] = 1.0
            print("[ OK ] %s (id=%s): %.3f (y=%.3f)" % (tank_name, tank_id, rating, y[i][0]))

    theta = 0.001 * numpy.random.rand(1, x.shape[1])
    print("[INFO] Theta shape: %r." % (theta.shape, ))

    alpha, lambda_, previous_cost = 0.001, 1.0, float("+inf")

    print("[INFO] Gradient descent.")
    for i in range(100000):
        theta_new = do_step(x, theta, y, r, lambda_, alpha)
        current_cost = cost(x, theta_new, y, r, lambda_)

        if i % 100 == 0 or i < 5:
            print("[INFO] Step #%d." % i)
            print("[INFO] Cost: %.6f (%.6f)." % (current_cost, previous_cost))
            print("[INFO] Alpha: %f." % alpha)

        if current_cost < previous_cost:
            alpha *= 1.05
            theta = theta_new
        elif abs(current_cost - previous_cost) < 1e-9:
            print("[WARN] Cost is not changing (step #%d)." % i)
            break
        else:
            alpha *= 0.5

        previous_cost = current_cost

    print("[ OK ] Theta: %r" % theta)
    print("[INFO] Cost: %.3f." % current_cost)


def cost(x, theta, y, r, lambda_):
    return (((x.dot(theta.T) - y) * r) ** 2).sum() / 2.0 + lambda_ * (theta ** 2).sum() / 2.0


def do_step(x, theta, y, r, lambda_, alpha):
    diff = (x.dot(theta.T) - y) * r
    theta_grad = diff.T.dot(x) + lambda_ * theta
    return theta - alpha * theta_grad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", dest="profile", help="learned profile", metavar="<profile.json>", required=True, type=argparse.FileType("rt"))
    parser.add_argument(dest="stats", help="user stats", metavar="<user.json>", type=argparse.FileType("rt"))
    main(parser.parse_args())
