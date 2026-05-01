#!/bin/bash
# 这个脚本把 WSL 的调用翻译给 Windows 的 Python

# 将所有传入的路径参数（如果包含 /mnt/ 格式）转换为 Windows 路径
ARGS=()
for arg in "$@"; do
    if [[ $arg == /mnt/* ]]; then
        ARGS+=("$(wslpath -w "$arg")")
    else
        ARGS+=("$arg")
    fi
done

# 调用 Windows 虚拟环境中的 python.exe
./.venv/Scripts/python.exe "${ARGS[@]}"