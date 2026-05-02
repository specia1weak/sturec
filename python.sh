#!/bin/bash

ARGS=()
for arg in "$@"; do
    if [[ $arg == /mnt/* ]]; then
        ARGS+=("$(wslpath -w "$arg")")
    else
        ARGS+=("$arg")
    fi
done

exec ./.venv/Scripts/python.exe "${ARGS[@]}"