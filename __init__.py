"""VulnFanatic-NG — LLM-assisted vulnerability research plugin for Binary Ninja.

This package is loaded by Binary Ninja as a UI plugin. On import it:
  1. registers the ``vulnfanatic.*`` settings group, and
  2. registers the VulnFanatic-NG sidebar widget (UI mode only).

The folder MUST be named ``vulnfanatic_ng`` (a valid Python identifier) because Binary
Ninja imports the plugin directory name as a module.
"""

from __future__ import annotations

from . import settings as _settings

# Always register settings, even when running headless (so scripts can read
# them and the values appear in the Settings dialog).
try:
    _settings.register_settings()
except Exception as exc:  # pragma: no cover - defensive
    import traceback
    print(f"[VulnFanatic-NG] Failed to register settings: {exc}")
    traceback.print_exc()

# Register the sidebar widget. This pulls in binaryninjaui (and therefore
# PySide6), which is only available when Binary Ninja runs with a UI. Failing
# softly keeps the plugin importable headlessly.
try:
    from .ui import sidebar as _sidebar  # noqa: F401  (import has side effects)

    _sidebar.register()
except ImportError:
    # No UI available (headless) — settings are still registered above.
    pass
except Exception as exc:  # pragma: no cover - defensive
    import traceback
    print(f"[VulnFanatic-NG] Failed to register sidebar widget: {exc}")
    traceback.print_exc()

__version__ = "1.0.0"
