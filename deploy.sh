#!/usr/bin/env sh
set -xeuo pipefail

git submodule init
git submodule update

deepmux upload
