# type: ignore

import os
import sys
import pathlib
import setuptools

from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension


__version__ = "0.0.1"


class get_pybind_include(object):

    """
    Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "numbits",
        sorted(["src/numbits.cpp"]),
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language="c++",
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(
    compiler,
    flagname,
) -> bool:

    """
    Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".cpp",
        delete=False,
    ) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):

    """
    Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """

    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


class BuildExt(build_ext):

    """
    A custom build extension for adding compiler-specific options.
    """

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": ["-O3", "-march=native", "-ffast-math"],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
install_requires = []
setup_requires = ["pybind11>=2.5.0"]


setup(
    name="numbits",
    version=__version__,
    description="Pack and unpack 1, 2 and 4 bit data to/from 8-bit numpy arrays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/telegraphic/numbits",
    author="Danny Price",
    author_email="dancpr@berkeley.edu",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_package_data=True,
    python_requires=">=3.5, <4",
    setup_requires=setup_requires,
    install_requires=install_requires,
    project_urls={
        "Source": "https://github.com/telegraphic/numbits",
        "Bug Reports": "https://github.com/telegraphic/numbits/issues",
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)