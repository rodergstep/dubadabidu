# SPDX-License-Identifier: AGPL-3.0-or-later
"""`python -m qc.tune_page` keeps working from a clone; the module is dubadabidu.lab.tune_page.

A compatibility entry point only: it is not in the wheel, and nothing imports it."""
import runpy

if __name__ == "__main__":
    runpy.run_module("dubadabidu.lab.tune_page", run_name="__main__", alter_sys=True)
