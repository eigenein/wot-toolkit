#!/usr/bin/env python3
# coding: utf-8

import argparse
import base64
import gzip
import logging
import sys

exec(base64.b64decode(b"QVBQTElDQVRJT05fSUQgPSAiOGVkMDBhZTYzMTlkNmQyYmI3ODYxNmNiNGJiNDg5OWQi"))

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stderr)

class GZipFileType:

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, filename):
        try:
            return gzip.open(filename, self.mode, compresslevel=6)
        except Exception as ex:
            raise argparse.ArgumentTypeError(str(ex)) from ex
