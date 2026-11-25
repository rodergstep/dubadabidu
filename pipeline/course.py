# SPDX-License-Identifier: AGPL-3.0-or-later
"""`python -m pipeline.course` keeps working from a clone; the module is dubadabidu.workflows.course.

A compatibility entry point only: it is not in the wheel, and nothing imports it."""
import runpy

if __name__ == "__main__":
    runpy.run_module("dubadabidu.workflows.course", run_name="__main__", alter_sys=True)
