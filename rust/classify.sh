#!/bin/bash
# Wrapper script for the image classifier
# Sets library path and runs the optimized binary

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export DYLD_LIBRARY_PATH="$SCRIPT_DIR/target/release"

exec "$SCRIPT_DIR/target/release/image_classifier" "$@"
