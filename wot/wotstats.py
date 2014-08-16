#!/usr/bin/env python3
# coding: utf-8

"WOTSTATS file interface."

import json
import struct


class Struct:
    "Compiled structs."
    file_header = struct.Struct("<QII")  # (magic, column_count, value_count)
    json_length = struct.Struct("<I")
    account_header = struct.Struct("<IIH")  # (magic, account_id, item_count)
    item = struct.Struct("<HII")  # (row, battles, wins)


class Magic:
    "Magic numbers."
    FILE_HEADER = 0x5354415453544f57  # WOTSTATS
    ACCOUNT = 0x3a434341  # ACC:


def write_header(output, column_count, value_count):
    "Writes database header."
    output.write(Struct.file_header.pack(Magic.FILE_HEADER, column_count, value_count))


def write_json(output, obj):
    "Writes serialized object to output."
    s = json.dumps(obj).encode("utf-8")
    output.write(Struct.json_length.pack(len(s)))
    output.write(s)


def write_account(output, account_id, items):
    "Writes account."
    assert account_id
    assert items

    item_count = len(items)
    output.write(Struct.account_header.pack(Magic.ACCOUNT, account_id, item_count))
    for row, battles, wins in items:
        output.write(Struct.item.pack(row, battles, wins))
    return item_count
