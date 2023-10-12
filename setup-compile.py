"""
Copyright (C) 2021 Lorenzo Meraldi

All rights reserved.
"""

import shutil
from pathlib import Path

# setuptools must come before distutils
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


class CopyInitBuildExt(build_ext):

    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        self.run_src_dir = Path(__file__).parent
        self.run_target_dir = build_dir if not self.inplace else self.run_src_dir

        top_level = Path("PROPHET_LOAD_LSTM")
        self.copy_init(top_level)
        self.copy_init(top_level / "main")
        self.copy_init(top_level / "util")

    def copy_init(self, package):
        init_file = Path(package) / "__init__.py"
        from_file = self.run_src_dir / init_file
        to_file = self.run_target_dir / init_file

        if not from_file.exists():
            raise RuntimeError(f"'{from_file}' not found")

        shutil.copyfile(str(from_file), str(to_file))


if __name__ == "__main__":
    setup(
        ext_modules=cythonize([
            Extension("PROPHET_LOAD_LSTM.main.ev_test", ["PROPHET_LOAD_LSTM/main/lstm_forecaster.py"]),
            Extension("PROPHET_LOAD_LSTM.util.util", ["PROPHET_LOAD_LSTM/util/util.py"])
        ],
            build_dir="build",
            language_level="3"),
        cmdclass=dict(
            build_ext=CopyInitBuildExt
        )
    )
