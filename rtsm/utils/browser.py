"""Cross-platform browser opener that handles WSL, native Windows, macOS, and Linux."""
from __future__ import annotations

import logging
import os
import subprocess
import webbrowser

logger = logging.getLogger(__name__)


def _is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def open_browser(url: str) -> None:
    """Open a URL in the default browser, handling WSL correctly.

    On WSL, `webbrowser.open()` fails because no Linux browser is installed.
    Instead, use `cmd.exe /c start` to open in the Windows host browser.
    """
    try:
        if _is_wsl():
            # WSL: use Windows cmd.exe to open in host browser
            # The URL needs empty string as first arg to `start` for URLs with special chars
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug(f"[browser] Opened {url} via cmd.exe (WSL)")
        else:
            webbrowser.open(url)
            logger.debug(f"[browser] Opened {url} via webbrowser")
    except Exception as e:
        # Never crash the server because browser didn't open
        logger.debug(f"[browser] Could not open {url}: {e}")
