#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import datetime

from recommonmark.transform import AutoStructify

import sphinx.ext.autodoc as autodoc

_original_get_doc = autodoc.Documenter.get_doc

def patched_get_doc(self, *args, **kwargs):
    try:
        if self.directive and hasattr(self.directive, "state"):
            doc = getattr(self.directive.state, "document", None)
            if doc and hasattr(doc, "settings") and not hasattr(doc.settings, "tab_width"):
                doc.settings.tab_width = 4
    except Exception:
        pass  # Silent fail-safe

    return _original_get_doc(self, *args, **kwargs)

autodoc.Documenter.get_doc = patched_get_doc


source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
master_doc = 'index'

breathe_projects = { "acados": "_build_doxygen_c_interface/xml/" }
breathe_default_project = "acados"

# -- Project information -----------------------------------------------------

project = 'acados'
now = datetime.datetime.now()
copyright = str(now.year) +', syscop'
author = 'syscop'
github_url = "https://github.com/acados/acados"

# https://stackoverflow.com/questions/62904172/how-do-i-replace-view-page-source-with-edit-on-github-links-in-sphinx-rtd-th
# These variables are intended to be set directly in the .rst files rather than in html_theme_options or conf.py in general. To make this actually apply site-wide as if it were defined in every .rst file, just put it in html_context.
# TODO: use rst files everywhere to get rid of this.
html_context = {
  'display_github': True,
  'github_user': 'acados',
  'github_repo': 'acados',
  'github_version': 'main/docs/',
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'breathe',
    'recommonmark',
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx_markdown_tables',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.youtube',
    'sphinxemoji.sphinxemoji',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'README.md', 'Thumbs.db', '.DS_Store', '*env*', 'requirements.txt', 'memory_management.md']

# -- GraphViz configuration --------------------------------------------------
graphviz_output_format = 'svg'

# -- Options for HTML output -------------------------------------------------

pygments_style = 'sphinx'
todo_include_todos = True

cpp_id_attributes = ['ACADOS_SYMBOL_EXPORT']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_logo = "_static/acados_logo.png"

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/acados/acados",
    "use_repository_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "show_navbar_depth": 1,  # 2 looks nice, but only extends for real subpages, like C interface, which is not so relevant
    "show_toc_level": 1,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_favicon = '_static/acados_favicon.png'

def setup(app):
    app.add_config_value('recommonmark_config', {
            'enable_auto_toc_tree': True,
            'auto_toc_tree_section': 'Contents',
            'enable_eval_rst': True,
            'enable_inline_math':True,
            'enable_math':True
            }, True)
    app.add_transform(AutoStructify)
