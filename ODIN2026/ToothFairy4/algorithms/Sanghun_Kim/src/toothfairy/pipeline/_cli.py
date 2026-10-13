"""Sub-command dispatch shared by the top-level ``main_*.py`` entry points.

Each entry point groups a few pipeline programs. The dispatcher takes the first argument
as the sub-command and hands **everything after it, verbatim** to that program's own
``main(argv)``, so ``toothfairy.cli.rft rollout --help`` prints the rollout program's help and
every flag documented for the module keeps working unchanged.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable


def dispatch(doc: str, commands: dict[str, tuple[Callable, str]],
             argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _usage(doc, commands)
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in commands:
        print(f"unknown command: {name}\n", file=sys.stderr)
        _usage(doc, commands, file=sys.stderr)
        return 2
    fn = commands[name][0]
    rc = asyncio.run(fn(rest)) if inspect.iscoroutinefunction(fn) else fn(rest)
    return int(rc or 0)


def _usage(doc: str, commands: dict[str, tuple[Callable, str]], file=sys.stdout) -> None:
    print(doc.strip(), file=file)
    width = max(len(k) for k in commands)
    print("\ncommands:", file=file)
    for name, (_, help_text) in commands.items():
        print(f"  {name:<{width}}  {help_text}", file=file)
    print("\nEvery command takes its own flags; add --help after it to see them.", file=file)
