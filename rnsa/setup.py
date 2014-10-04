#!/usr/bin/env python3
# coding: utf-8

from distutils.core import setup, Extension


rnsa = Extension(
    "rnsa",
    sources=["rnsa.c"],
    extra_compile_args=["-O3", "-std=c99"],
)

setup(
    name="rnsa",
    version="1.0",
    description="Refined Neighbor Selection Algorithm extension.",
    ext_modules=[rnsa],
)
