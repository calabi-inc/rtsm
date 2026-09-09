"""Make the flat example modules importable when running pytest from anywhere,
and keep the suite off real SDL.

EstopMonitor.start() calls pygame.init() on the calling thread. On a box where
SDL's device enumeration hangs (seen 2026-09-08: Windows 11 26200, no controller
attached, a bare ``import pygame; pygame.init()`` never returns) every
``create_app()`` test hangs with it. The server tests exercise the agent, not
the pad, so a minimal in-process fake pygame is installed for the whole session
unless ``RC_CAR_TESTS_REAL_GAMEPAD=1``. It exposes exactly the surface
estop.py touches and reports zero joysticks. test_estop.py installs its own
richer fake per test on top of this one (monkeypatch.setitem on sys.modules),
so those tests are unaffected.
"""
import os
import sys
import types
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _fake_pygame() -> types.ModuleType:
    pg = types.ModuleType("pygame")
    state = {"init": False}
    pg.JOYDEVICEADDED = 1541
    pg.JOYDEVICEREMOVED = 1542
    pg.get_init = lambda: state["init"]

    def _init():
        state["init"] = True
        return (0, 0)

    pg.init = _init
    pg.quit = lambda: state.update(init=False)

    joystick = types.ModuleType("pygame.joystick")
    joystick.init = lambda: None
    joystick.quit = lambda: None
    joystick.get_count = lambda: 0

    def _no_joystick(_i):
        raise RuntimeError("fake pygame: no joysticks")

    joystick.Joystick = _no_joystick

    event = types.ModuleType("pygame.event")
    event.pump = lambda: None
    event.get = lambda *_a, **_k: []
    event.clear = lambda *_a, **_k: None

    pg.joystick = joystick
    pg.event = event
    sys.modules["pygame.joystick"] = joystick
    sys.modules["pygame.event"] = event
    return pg


if os.environ.get("RC_CAR_TESTS_REAL_GAMEPAD") != "1" and "pygame" not in sys.modules:
    sys.modules["pygame"] = _fake_pygame()
