#!/usr/bin/env python3
# coding: utf-8

import struct


magic = b"WOTSTATS3"
entry = struct.Struct("<IHII")  # (account_id, tank_id, battles, wins)
