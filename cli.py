"""
cli.py
======
CLI interface for real-time control of Vol-adaptive Strategy.

Provides:
- Command-line interface: python -m src.cli [command] [args]
- Interactive REPL: python -m src.cli
- Pause/resume trading
- Update strategy configuration
- View current state

Uses socket server listening for JSON commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
from datetime import datetime
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Command Types
# ============================================================================

class Command:
    """Base for all CLI commands."""

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        """Execute command and return result."""
        raise NotImplementedError


class PauseCommand(Command):
    """Pause trading."""

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        handler.pause_trading()
        return {"status": "success", "message": "Trading paused"}


class ResumeCommand(Command):
    """Resume trading."""

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        handler.resume_trading()
        return {"status": "success", "message": "Trading resumed"}


class StatusCommand(Command):
    """Get current trading status."""

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        return handler.get_status()


class ConfigCommand(Command):
    """Update configuration."""

    def __init__(self, updates: dict[str, Any]):
        self.updates = updates

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        return handler.update_config(self.updates)


class PositionCommand(Command):
    """Get current positions."""

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        return handler.get_positions()


class SignalCommand(Command):
    """Get current signal."""

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        return handler.get_signal()


class GetCommand(Command):
    """Get configuration."""

    def __init__(self, keys: list[str] | None = None):
        self.keys = keys or []

    def execute(self, handler: CliCommandHandler) -> dict[str, Any]:
        return handler.get_config(self.keys)


# ============================================================================
# CLI Command Handler
# ============================================================================

class CliCommandHandler:
    """
    Handles CLI commands and manages trading state.

    This is the interface between the CLI socket server and the orchestrator.
    """

    def __init__(self):
        self.is_paused = False
        self.callbacks: dict[str, Callable] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for events.

        Events:
        - pause: when trading is paused
        - resume: when trading is resumed
        - config_update: when config is updated {key: value}
        """
        self.callbacks[event] = callback

    def pause_trading(self) -> None:
        """Pause trading."""
        self.is_paused = True
        logger.info("Trading paused via CLI")
        self._fire_callback("pause")

    def resume_trading(self) -> None:
        """Resume trading."""
        self.is_paused = False
        logger.info("Trading resumed via CLI")
        self._fire_callback("resume")

    def get_status(self) -> dict[str, Any]:
        """Get trading status."""
        return {
            "is_paused": self.is_paused,
            "timestamp": str(datetime.now()),
        }

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update configuration."""
        logger.info(f"Config update requested: {updates}")
        self._fire_callback("config_update", updates)
        return {"status": "success", "message": "Config updated", "updates": updates}

    def get_positions(self) -> dict[str, Any]:
        """Get current positions (requires callback to orchestrator)."""
        return self._fire_callback("get_positions") or {
            "positions": {},
            "total_value": 0.0,
        }

    def get_signal(self) -> dict[str, Any]:
        """Get current signal (requires callback to orchestrator)."""
        return self._fire_callback("get_signal") or {
            "signal": None,
            "confidence": 0.0,
        }

    def get_config(self, keys: list[str] | None = None) -> dict[str, Any]:
        """Get configuration (entire or specific keys with dot notation)."""
        return self._fire_callback("get_config", keys) or {
            "config": {},
        }

    def _fire_callback(self, event: str, *args: Any) -> Any:
        """Fire callback if registered."""
        if event in self.callbacks:
            return self.callbacks[event](*args)
        return None


# ============================================================================
# Socket Server
# ============================================================================

class CliSocketServer:
    """
    TCP socket server for CLI commands.

    Listens for JSON commands on localhost:port and returns JSON responses.

    Protocol:
    -> {"command": "pause"}
    <- {"status": "success", "message": "Trading paused"}

    -> {"command": "status"}
    <- {"is_paused": false, "timestamp": "..."}

    -> {"command": "config", "updates": {"target_vol": 0.10}}
    <- {"status": "success", "message": "Config updated", ...}
    """

    def __init__(
        self,
        host: str,
        port: int,
        handler: CliCommandHandler,
        timeout: int = 30,
    ):
        self.host = host
        self.port = port
        self.handler = handler
        self.timeout = timeout
        self.running = False
        self.server_socket: Optional[socket.socket] = None

    def start(self) -> None:
        """Start socket server in background thread."""
        self.running = True
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        logger.info(f"CLI socket server listening on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop socket server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("CLI socket server stopped")

    def _run(self) -> None:
        """Main server loop (runs in background thread)."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Windows-compatible socket configuration
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Allow socket to close immediately without TIME_WAIT
            import struct
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, 
                                         struct.pack('ii', 1, 0))
            
            # On Windows, use SO_EXCLUSIVEADDRUSE to avoid port conflicts
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                try:
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                except OSError:
                    pass  # Not available on all systems, that's ok
            
            for attempt in range(3):
                try:
                    self.server_socket.bind((self.host, self.port))
                    logger.info(f"CLI socket server bound to {self.host}:{self.port}")
                    break
                except OSError as e:
                    if attempt == 2:
                        logger.error(f"Failed to bind after {attempt + 1} attempts: {e}")
                        raise
                    logger.info(f"Bind attempt {attempt + 1} failed, retrying... ({e})")
                    import time
                    time.sleep(0.5)
            
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # Brief timeout for shutdown check

            while self.running:
                try:
                    client, addr = self.server_socket.accept()
                    client.settimeout(self.timeout)
                    self._handle_client(client, addr)
                except socket.timeout:
                    continue

        except Exception as e:
            logger.error(f"CLI socket server error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass

    def _handle_client(self, client: socket.socket, addr: tuple) -> None:
        """Handle single client connection."""
        try:
            # Receive command
            data = client.recv(4096)
            if not data:
                return

            logger.debug(f"Received {len(data)} bytes: {data[:50]}")  # Log first 50 bytes for debugging

            # Decode with error handling - try UTF-8 first, then handle BOM
            try:
                cmd_str = data.decode('utf-8')
            except UnicodeDecodeError:
                # Try UTF-8 with BOM handling
                try:
                    cmd_str = data.decode('utf-8-sig')
                except UnicodeDecodeError:
                    # Try UTF-16LE (common on Windows)
                    try:
                        cmd_str = data.decode('utf-16-le')
                    except UnicodeDecodeError:
                        # Fall back to UTF-16 
                        cmd_str = data.decode('utf-16')
            
            logger.debug(f"Decoded command: {cmd_str}")
            cmd_dict = json.loads(cmd_str)
            command = self._parse_command(cmd_dict)

            if command:
                result = command.execute(self.handler)
                response = json.dumps(result).encode('utf-8')
                logger.debug(f"Sending response: {result}")
                client.sendall(response)
            else:
                error = {"status": "error", "message": "Unknown command"}
                client.sendall(json.dumps(error).encode('utf-8'))

        except json.JSONDecodeError as e:
            error = {"status": "error", "message": f"Invalid JSON: {e}"}
            client.sendall(json.dumps(error).encode('utf-8'))
        except UnicodeDecodeError as e:
            error = {"status": "error", "message": f"Encoding error: {e}. Data: {data[:20]}"} # type: ignore
            try:
                client.sendall(json.dumps(error).encode('utf-8'))
            except:
                pass
        except Exception as e:
            logger.error(f"Client handler error: {e}", exc_info=True)
            try:
                error = {"status": "error", "message": str(e)}
                client.sendall(json.dumps(error).encode('utf-8'))
            except:
                pass
        finally:
            client.close()

    @staticmethod
    def _parse_command(cmd_dict: dict[str, Any]) -> Optional[Command]:
        """Parse JSON command dict to Command object."""
        cmd_name = cmd_dict.get("command", "").lower()

        if cmd_name == "pause":
            return PauseCommand()
        elif cmd_name == "resume":
            return ResumeCommand()
        elif cmd_name == "status":
            return StatusCommand()
        elif cmd_name == "config":
            updates = cmd_dict.get("updates", {})
            return ConfigCommand(updates)
        elif cmd_name == "positions":
            return PositionCommand()
        elif cmd_name == "signal":
            return SignalCommand()
        elif cmd_name == "get":
            keys = cmd_dict.get("keys", [])
            return GetCommand(keys)
        else:
            return None


# ============================================================================
# CLI Client (for testing)
# ============================================================================

class CliClient:
    """Simple client for testing CLI commands."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port

    def send_command(self, command: dict) -> dict[str, Any]:
        """Send command and get response."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            sock.sendall(json.dumps(command).encode('utf-8'))

            response = sock.recv(4096).decode('utf-8')
            sock.close()

            return json.loads(response)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause(self) -> dict:
        return self.send_command({"command": "pause"})

    def resume(self) -> dict:
        return self.send_command({"command": "resume"})

    def status(self) -> dict:
        return self.send_command({"command": "status"})

    def update_config(self, updates: dict) -> dict:
        return self.send_command({"command": "config", "updates": updates})

    def get_positions(self) -> dict:
        return self.send_command({"command": "positions"})

    def get_signal(self) -> dict:
        return self.send_command({"command": "signal"})

    def get_config(self, keys: list[str] | None = None) -> dict:
        """Retrieve config (entire or specific keys with dot notation)."""
        return self.send_command({"command": "get", "keys": keys or []})


def _parse_config_updates(updates_list: list[str]) -> dict[str, Any]:
    """
    Parse config updates with dot notation support.
    
    Examples:
        target_vol=0.12 -> {"target_vol": 0.12}
        regime_detector.n_init=20 -> {"regime_detector": {"n_init": 20}}
        strategy.regime_weights.CALM.SPY=2.0 -> {"strategy": {"regime_weights": {"CALM": {"SPY": 2.0}}}}
    """
    result = {}
    
    for update in updates_list:
        if "=" not in update:
            continue
        
        key_path, value_str = update.split("=", 1)
        keys = key_path.split(".")
        
        # Type conversion
        try:
            if value_str.lower() in ("true", "false"):
                value = value_str.lower() == "true"
            else:
                value = float(value_str)
        except ValueError:
            value = value_str
        
        # Build nested dict
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return result

def _execute_command(client: CliClient, args: argparse.Namespace) -> None:
    """Execute command based on parsed arguments."""
    result = None
    
    if args.command == "pause":
        result = client.pause()
    elif args.command == "resume":
        result = client.resume()
    elif args.command == "status":
        result = client.status()
    elif args.command == "positions":
        result = client.get_positions()
    elif args.command == "signal":
        result = client.get_signal()
    elif args.command == "config":
        updates = _parse_config_updates(args.updates)
        result = client.update_config(updates)
    elif args.command == "get":
        result = client.get_config(args.keys if hasattr(args, 'keys') else [])
    
    if result:
        print(json.dumps(result, indent=2))


def _create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Vol-adaptive Strategy CLI - Control trading via command-line or REPL",
        prog="python -m src.cli",
        add_help=True
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Pause command
    subparsers.add_parser("pause", help="Pause trading")
    
    # Resume command
    subparsers.add_parser("resume", help="Resume trading")
    
    # Status command
    subparsers.add_parser("status", help="Get trading status")
    
    # Positions command
    subparsers.add_parser("positions", help="View current positions")
    
    # Signal command
    subparsers.add_parser("signal", help="Get current regime and confidence")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Update strategy configuration")
    config_parser.add_argument(
        "updates",
        nargs="*",
        metavar="KEY=VALUE",
        help="Config updates (e.g., target_vol=0.12 regime_detector.n_init=20)"
    )
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Retrieve configuration (entire or specific keys)")
    get_parser.add_argument(
        "keys",
        nargs="*",
        metavar="KEY",
        help="Config keys to retrieve (dot notation, e.g., regime_detector.n_init). Omit for entire config."
    )
    
    return parser


if __name__ == "__main__":
    # Default connection settings
    host = "127.0.0.1"
    port = 9000
    client = CliClient(host=host, port=port)
    
    # Create parser
    parser = _create_parser()
    args = parser.parse_args()
    
    # Command-line mode
    if args.command:
        _execute_command(client, args)
    
    # REPL mode (no command specified)
    else:
        print("=" * 60)
        print("Vol-adaptive Strategy - CLI REPL")
        print("=" * 60)
        print(f"Connected to {host}:{port}")
        print("\nType 'help' or 'h' for available commands\n")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("cli> ").strip()
                
                if not user_input:
                    continue
                
                # Special REPL commands
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                
                elif user_input.lower() in ("help", "h"):
                    parser.print_help()
                    continue
                
                # Parse and execute as if it were a command-line argument
                try:
                    repl_args = parser.parse_args(user_input.split())
                    if repl_args.command:
                        _execute_command(client, repl_args)
                except SystemExit:
                    # argparse calls sys.exit on error, catch it gracefully
                    continue
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
