#!/usr/bin/env python3
"""
control.py
----------
CLI for the VIX strategy trader control socket.

Usage examples:

  # One-shot commands:
  python control.py status
  python control.py get
  python control.py get ema_span atr_multiplier
  python control.py set band_std_dev 0.9
  python control.py set band_std_dev 0.9 --persist
  python control.py reload-config
  python control.py pause
  python control.py resume
  python control.py close-position

  # Interactive REPL:
  python control.py repl

  # Custom host/port:
  python control.py --host 127.0.0.1 --port 9999 status

All commands print the server JSON response (pretty-printed).
"""

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9999
DEFAULT_CONFIG = "config.yaml"

REPL_ALIASES: dict[str, str] = {
    "reload": "reload-config",
    "close": "close-position",
}

REPL_LOCAL_HELP = (
    "\nLocal REPL commands:\n"
    "  help                - show this message\n"
    "  clear               - clear the screen\n"
    "  quit / exit / Ctrl-D - exit REPL\n"
    "Aliases:\n"
    "  reload -> reload-config\n"
    "  close  -> close-position\n"
)


def send_command(host: str, port: int, msg: dict[str, Any]) -> dict[str, Any]:
    """Open a TCP connection, send one JSON command, return parsed reply."""
    raw = (json.dumps(msg) + "\n").encode()
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(raw)
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
    line = buf.split(b"\n")[0]
    return json.loads(line.decode())


def pretty(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def persist_strategy_param(config_path: str, key: str, value: Any) -> None:
    """
    Persist one key in the strategy section of config.yaml.
    Updates one key line if present, otherwise inserts it into the strategy block.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"Config file is empty: {config_path}")

    strategy_idx: int | None = None
    strategy_indent = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)strategy\s*:\s*(?:#.*)?$", line.rstrip("\r\n"))
        if m:
            strategy_idx = i
            strategy_indent = len(m.group(1))
            break
    if strategy_idx is None:
        raise ValueError("Could not find 'strategy:' section in config file")

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    block_end = len(lines)
    for i in range(strategy_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if indent_of(lines[i]) <= strategy_indent:
            block_end = i
            break

    child_indent: int | None = None
    for i in range(strategy_idx + 1, block_end):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        ind = indent_of(lines[i])
        if ind > strategy_indent:
            child_indent = ind
            break
    if child_indent is None:
        child_indent = strategy_indent + 2

    key_re = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*(.*?)(\s+#.*)?(\r?\n?)$")
    for i in range(strategy_idx + 1, block_end):
        line = lines[i]
        if line.lstrip().startswith("#"):
            continue
        m = key_re.match(line)
        if m and len(m.group(1)) > strategy_indent:
            comment = m.group(3) or ""
            newline = m.group(4) or "\n"
            lines[i] = f"{' ' * len(m.group(1))}{key}: {_yaml_scalar(value)}{comment}{newline}"
            path.write_text("".join(lines), encoding="utf-8")
            return

    newline = "\n"
    if lines[0].endswith("\r\n"):
        newline = "\r\n"
    lines.insert(block_end, f"{' ' * child_indent}{key}: {_yaml_scalar(value)}{newline}")
    path.write_text("".join(lines), encoding="utf-8")


def filter_params_reply(reply: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    if not keys:
        return reply

    params = reply.get("params")
    if not reply.get("ok") or not isinstance(params, dict):
        return reply

    selected = {k: params[k] for k in keys if k in params}
    missing = [k for k in keys if k not in params]

    result = dict(reply)
    result["params"] = selected
    if missing:
        result["missing"] = missing
    return result


def add_remote_command_parsers(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("status", help="Show current trader state")

    subparsers.add_parser("reload-config", help="Reload config.yaml + re-warm-up")
    subparsers.add_parser("pause", help="Pause trading (keep position)")
    subparsers.add_parser("resume", help="Resume trading")
    subparsers.add_parser("close-position", help="Flatten open position")

    gp = subparsers.add_parser("get", help="Dump strategy parameters")
    gp.add_argument("keys", nargs="*", help="Optional specific keys to dump")

    sp = subparsers.add_parser("set", help="Hot-update a single parameter")
    sp.add_argument("key", help="Parameter name (e.g. band_std_dev)")
    sp.add_argument("value", help="New value (will be cast to correct type)")
    sp.add_argument("--persist", action="store_true", help="Also save to config.yaml")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control tool for the VIX strategy trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to config YAML used by --persist (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command")
    add_remote_command_parsers(subparsers)
    subparsers.add_parser("repl", help="Interactive REPL")
    return parser


def build_repl_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command")
    add_remote_command_parsers(subparsers)
    return parser


def resolve_remote_command(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], bool, str | None, Any]:
    """Map parsed argparse namespace to server message and local post-actions."""
    command = args.command

    if command == "status":
        return ({"cmd": "status"}, [], False, None, None)
    if command == "reload-config":
        return ({"cmd": "reload_config"}, [], False, None, None)
    if command == "pause":
        return ({"cmd": "pause"}, [], False, None, None)
    if command == "resume":
        return ({"cmd": "resume"}, [], False, None, None)
    if command == "close-position":
        return ({"cmd": "close_position"}, [], False, None, None)
    if command == "get":
        return ({"cmd": "get_params"}, list(args.keys or []), False, None, None)
    if command == "set":
        return (
            {"cmd": "set_param", "key": args.key, "value": args.value},
            [],
            bool(args.persist),
            str(args.key),
            args.value,
        )

    raise ValueError(f"Unsupported command: {command}")


def execute_remote_command(
    host: str,
    port: int,
    config_path: str,
    msg: dict[str, Any],
    selected_keys: list[str],
    persist: bool,
    persist_key: str | None,
    persist_fallback_value: Any,
) -> int:
    try:
        reply = send_command(host, port, msg)
        reply = filter_params_reply(reply, selected_keys)
        print(pretty(reply))

        if persist:
            if reply.get("ok"):
                if not persist_key:
                    raise ValueError("Persist key missing for set --persist")
                persist_strategy_param(
                    config_path,
                    persist_key,
                    reply.get("value", persist_fallback_value),
                )
                print(f"[info] Persisted to {config_path}")
            else:
                print("[warn] Not persisted because set command failed.")

        return 0 if reply.get("ok") else 1

    except ConnectionRefusedError:
        print(f"[error] Cannot connect to {host}:{port}. Is trader.py running?")
        return 1
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


def run_repl(host: str, port: int, config_path: str) -> None:
    cmd_parser = build_repl_command_parser()
    print(f"Connected to trader at {host}:{port}  (type 'help' for commands)\n")

    while True:
        try:
            line = input("trader> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        tokens = line.split()
        local_cmd = tokens[0].lower()

        if local_cmd in ("quit", "exit"):
            break
        if local_cmd == "help":
            print(cmd_parser.format_help().rstrip())
            print(REPL_LOCAL_HELP)
            continue
        if local_cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        tokens[0] = REPL_ALIASES.get(local_cmd, local_cmd)

        try:
            args = cmd_parser.parse_args(tokens)
        except SystemExit:
            continue

        if args.command is None:
            print("Unknown command. Type 'help'.")
            continue

        msg, selected_keys, persist, persist_key, persist_fallback_value = resolve_remote_command(args)
        execute_remote_command(
            host,
            port,
            config_path,
            msg,
            selected_keys,
            persist,
            persist_key,
            persist_fallback_value,
        )


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "repl":
        run_repl(args.host, args.port, args.config)
        return

    msg, selected_keys, persist, persist_key, persist_fallback_value = resolve_remote_command(args)
    code = execute_remote_command(
        args.host,
        args.port,
        args.config,
        msg,
        selected_keys,
        persist,
        persist_key,
        persist_fallback_value,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
