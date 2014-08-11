#!/usr/bin/env python3
# coding: utf-8

from distutils.core import setup, Extension


trainer = Extension("trainer", sources=["trainer.c"], extra_compile_args=["-O3", "-funroll-loops"])

setup(
    name="trainer",
    version="1.0",
    description="Collaboration filtering trainer module.",
    ext_modules=[trainer],
)
