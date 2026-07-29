# Scripts

`validate_generation_artifact.py` and `validate_notebooks.py` are the supported command-line checks.

The `build_*_notebook.py` files are the source generators used to produce the corresponding notebooks
during development. The notebooks themselves are the runnable, frozen research implementation and should
be used for reproduction. The builders are retained so maintainers can trace how the notebook artifacts
were constructed; they are not required for a paper run.
