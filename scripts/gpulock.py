#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Runs a command while holding the GPU benchmark lock, so tests never overlap a measurement.

    scripts/gpulock.py -- swift test -c release
"""
import fcntl
import subprocess
import sys

LOCK_PATH = "/tmp/gemma-embed-gpu.lock"

argv = sys.argv[1:]
if argv and argv[0] == "--":
    argv = argv[1:]
if not argv:
    sys.exit("usage: gpulock.py -- command [args...]")
with open(LOCK_PATH, "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    sys.exit(subprocess.call(argv))
