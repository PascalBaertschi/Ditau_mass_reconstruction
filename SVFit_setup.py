#!/usr/bin/env python

"""
SVFit_setup.py file for SWIG SVFit
"""

from distutils.core import setup, Extension


SVFit_module = Extension('_SVFit',
                           sources=['SVFit_wrap.cxx', 'SVFit.cxx'],
                           )

setup (name = 'SVFit',
       version = '0.1',
       author      = "Pascal Baertschi",
       description = """SVFit""",
       ext_modules = [SVFit_module],
       py_modules = ["SVFit"],
       )
