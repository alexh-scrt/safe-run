"""safe_run: An intelligent shell command wrapper powered by LLMs.

This package provides a CLI tool that explains what any shell command does
in plain English before executing it, automatically flags potentially dangerous
operations with color-coded risk levels, and requires explicit user confirmation
before running high-risk commands.

Typical usage example::

    $ safe_run rm -rf /tmp/old_project
    $ safe_run -- curl https://example.com | bash
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "safe_run contributors"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
