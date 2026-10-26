Building and hosting the documentation
======================================

Local build
-----------

From the repository root, install the documentation extra and build HTML:

.. code-block:: sh

   python -m pip install -e ".[docs]"
   python -m sphinx -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html`` in your browser. The documentation
dependencies require Python 3.11 or later. API pages import the installed
package, including its runtime dependencies; keep the editable installation
to reflect local docstring edits. Warnings fail the build.

The API reference uses ``sphinx.ext.autodoc`` to read docstrings and
``sphinx.ext.napoleon`` to parse the NumPy convention. When adding a public
export, add its directive to ``api.rst``.

Read the Docs
-------------

The repository's ``.readthedocs.yaml`` selects Python 3.12 on Ubuntu 24.04,
installs the package with its ``docs`` extra, and builds ``docs/conf.py``
with warnings treated as errors.

To activate hosting:

#. Commit and push the documentation files to the repository you intend to host.
#. Sign in to Read the Docs and import that repository as a project. For a
   private repository, use a Read the Docs plan that supports private repositories.
#. Select the intended default branch and trigger the first build. Ensure
   documentation visibility matches the repository's intended audience.
#. Check the build log and open the documentation URL shown by Read the Docs.
   Verify the repository webhook is connected for subsequent builds.

Project registration, repository access and publishing are account-side steps;
adding the configuration file alone does not activate hosting.

Configuration details are described in the
`Read the Docs configuration reference
<https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_.
NumPy docstring support is described in the
`Sphinx Napoleon documentation
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.
