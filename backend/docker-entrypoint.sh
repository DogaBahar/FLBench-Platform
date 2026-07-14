#!/bin/sh
set -e

python -m scripts.migrate

exec "$@"
