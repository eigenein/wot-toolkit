#!/usr/bin/env python3
# coding: utf-8

"WOTSTATS file interface."

import json
import struct


class Struct:
    "Compiled structs."
    file_header = struct.Struct("<QII")  # (magic, account_count, item_count)
    json_header = struct.Struct("<II")  # (magic, length)
    account_header = struct.Struct("<IIH")  # (magic, account_id, item_count)
    item = struct.Struct("<HII")  # (row, battles, wins)


class Magic:
    "Magic numbers."
    FILE_HEADER = 0x5354415453544f57  # WOTSTATS
    ACCOUNT = 0x3a434341  # ACC:
    JSON = 0x4e4f534a  # JSON


def write_header(fp, account_count, item_count):
    "Writes database header."
    fp.write(Struct.file_header.pack(Magic.FILE_HEADER, account_count, item_count))


def read_header(fp):
    "Reads database header."
    magic, account_count, item_count = Struct.file_header.unpack(fp.read(Struct.file_header.size))
    assert magic == Magic.FILE_HEADER
    return account_count, item_count


def write_json(fp, obj):
    "Writes serialized object."
    s = json.dumps(obj).encode("utf-8")
    fp.write(Struct.json_header.pack(Magic.JSON, len(s)))
    fp.write(s)


def read_json(fp):
    "Reads JSON object from database."
    magic, length = Struct.json_header.unpack(fp.read(Struct.json_header.size))
    assert magic == Magic.JSON
    return fp.read(length).decode("utf-8")


def write_account(fp, account_id, items):
    "Writes account data."
    assert account_id
    assert items

    item_count = len(items)
    fp.write(Struct.account_header.pack(Magic.ACCOUNT, account_id, item_count))
    for row, battles, wins in items:
        fp.write(Struct.item.pack(row, battles, wins))
    return item_count


def read_account(fp):
    "Reads account data."
    magic, account_id, item_count = Struct.account_header.unpack(fp.read(Struct.account_header.size))
    assert magic == Magic.ACCOUNT
    items = [Struct.item.unpack(fp.read(Struct.item.size)) for _ in range(item_count)]
    return account_id, items


def read(fp):
    "High-level wotstats file reading."
    account_count, item_count = read_header(fp)
    encyclopedia = read_json(fp)
    accounts = (read_account(fp) for _ in range(account_count))
    return account_count, item_count, encyclopedia, accounts
