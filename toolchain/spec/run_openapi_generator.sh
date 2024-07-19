#!/bin/bash

set -x

PYTHONPATH=../../../oss-ops:../.. python3 -m toolchain.spec.generate
