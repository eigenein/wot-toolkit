#!/usr/bin/env python3
# coding: utf-8

import argparse
import csv
import gzip


class CsvReaderGZipFileType:

    def __call__(self, filename):
        try:
            return csv.reader(gzip.open(filename, "rt", encoding="utf-8"))
        except Exception as ex:
            raise argparse.ArgumentTypeError(str(ex)) from ex


class CsvWriterGZipFileType:

    def __call__(self, filename):
        try:
            return csv.writer(gzip.open(filename, "wt", compresslevel=6, encoding="utf-8"))
        except Exception as ex:
            raise argparse.ArgumentTypeError(str(ex)) from ex
