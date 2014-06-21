#!/usr/bin/env python3
# coding: utf-8

import sys; sys.dont_write_bytecode = True

import argparse

import utils


def main(args):
    shared_header = None
    for reader in args.readers:
        print("[INFO] Reading from reader.")

        header = next(reader)
        print("[ OK ] Read header.")

        if shared_header:
            if header != shared_header:
                raise ValueError("header mismatch")
            print("[ OK ] Validated header.")
        else:
            shared_header = header
            args.writer.writerow(header)
            print("[ OK ] Written header.")
        for i, row in enumerate(reader, start=1):
            args.writer.writerow(row)
            if i % 1000 == 1:
                print("[INFO] Row #%d." % i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="readers", help="input file", metavar="<stats.csv.gz>", nargs=argparse.ONE_OR_MORE, type=utils.CsvReaderGZipFileType())
    parser.add_argument("-o", dest="writer", help="output file", metavar="<output.csv.gz>", type=utils.CsvWriterGZipFileType())
    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        pass
