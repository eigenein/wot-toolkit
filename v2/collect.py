#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse
import itertools
import json
import logging
import pathlib
import random
import time

import requests

import shared


LIMIT = 100
SLEEP_TIME = [None, 1.0, 60.0, 600.0, 3600.0]


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistics robot.")
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
