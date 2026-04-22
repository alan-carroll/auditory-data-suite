"""Runtime compatibility helpers for third-party package quirks."""

from __future__ import annotations

from functools import total_ordering
import sys
import types

from packaging.version import Version


@total_ordering
class _LooseVersion:
    """
    Minimal replacement for distutils.version.LooseVersion.

    kivy_garden.matplotlib still imports this on Python 3.12+, where
    stdlib distutils no longer exists. The backend only uses it for
    straightforward version comparisons, which packaging.Version covers.
    """

    def __init__(self, value):
        self._raw = str(value)
        self._parsed = Version(self._raw)

    def _coerce(self, other):
        return other if isinstance(other, _LooseVersion) else _LooseVersion(other)

    def __eq__(self, other):
        other = self._coerce(other)
        return self._parsed == other._parsed

    def __lt__(self, other):
        other = self._coerce(other)
        return self._parsed < other._parsed

    def __repr__(self):
        return f"_LooseVersion('{self._raw}')"

    def __str__(self):
        return self._raw


def install_distutils_version_shim():
    """
    Provide distutils.version.LooseVersion on Python 3.12+.

    If a real distutils module exists, leave it alone.
    """
    try:
        import distutils.version  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    distutils_mod = sys.modules.setdefault("distutils", types.ModuleType("distutils"))
    version_mod = types.ModuleType("distutils.version")
    version_mod.LooseVersion = _LooseVersion
    distutils_mod.version = version_mod
    sys.modules["distutils.version"] = version_mod
