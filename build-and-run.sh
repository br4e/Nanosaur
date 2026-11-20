#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Xcode 27 rejects the 10.13 deployment target in CMakeLists.txt.
cmake -S . -B build -G Xcode -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0
cmake --build build --config RelWithDebInfo -- -j "$(sysctl -n hw.ncpu)" -quiet

open build/RelWithDebInfo/Nanosaur.app
