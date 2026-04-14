"""
Shared CLI helpers for the colorama-heavy entry points.

Centralizes color styling and the repeat-until-valid input loops so
callers read as `warn("...")` / `ask_yes_no("...")` instead of walls of
Style.BRIGHT + Fore.X + ... + Style.RESET_ALL.
"""
import logging
import colorama
from colorama import Fore, Back, Style

colorama.init()  # idempotent; safe even if an entry point also calls it

# -- colored output -------------------------------------------------------

def cprint(msg, *, fg=None, bg=None, bright=True, **print_kwargs):
    """Print `msg` wrapped in colorama styling, always resetting after."""
    prefix = ""
    if bright:
        prefix += Style.BRIGHT
    if fg:
        prefix += fg
    if bg:
        prefix += bg
    print(f"{prefix}{msg}{Style.RESET_ALL}", **print_kwargs)

def info(msg, **kw):    cprint(msg, fg=Fore.CYAN, **kw)
def note(msg, **kw):    cprint(msg, fg=Fore.MAGENTA, **kw)
def warn(msg, **kw):    cprint(msg, fg=Fore.YELLOW, **kw)
def error(msg, **kw):   cprint(msg, fg=Fore.RED, **kw)
def success(msg, **kw): cprint(msg, fg=Fore.GREEN, **kw)
def banner(msg, **kw):  cprint(msg, bg=Back.GREEN, **kw)

def fail(exc, msg):
    """
    `logging.exception` + red console message. For except-blocks in
    interactive flows where the user should see something but the
    traceback belongs in the log.
    """
    logging.exception(exc)
    error(msg)

# -- validated prompts -----------------------------------------------------

def ask_choice(prompt, choices):
    """
    Re-prompt until the user enters one of `choices` (compared after
    .strip().lower()). Returns the matched choice string.
    """
    valid = tuple(choices)
    while (resp := input(prompt).strip().lower()) not in valid:
        continue
    return resp

def ask_yes_no(prompt):
    """Re-prompt until y/n; returns True for 'y'."""
    return ask_choice(prompt, ("y", "n")) == "y"