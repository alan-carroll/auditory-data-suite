"""Shared logging setup for CLI and GUI entry points."""
import logging
import sys
import threading


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_file_logging(path, *, level=logging.ERROR, mode="w"):
    """
    Route root-logger output to one file and drop the noisy defaults.

    Call this early in an entry point so third-party imports inherit the
    quieter policy for this process.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level)

    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    return root


def install_excepthooks(logger=None):
    """Log otherwise-uncaught exceptions to the configured log file."""
    target = logger or logging.getLogger("uncaught")

    def _handle(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        target.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _thread_handle(args):
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        target.critical(
            "Uncaught exception in thread %s",
            args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _handle
    threading.excepthook = _thread_handle


def install_tkinter_cleanup_guard():
    """Suppress noisy Tk image finalizer errors after GUI shutdown."""
    try:
        import tkinter as tk
    except ImportError:
        return

    original_del = getattr(tk.Image, "__del__", None)
    if original_del is None or getattr(original_del, "_ads_guarded", False):
        return

    def _guarded_del(self):
        try:
            original_del(self)
        except RuntimeError as exc:
            message = str(exc)
            if (
                    "main thread is not in main loop" in message
                    or "application has been destroyed" in message):
                return
            raise

    _guarded_del._ads_guarded = True
    _guarded_del._ads_original = original_del
    tk.Image.__del__ = _guarded_del
