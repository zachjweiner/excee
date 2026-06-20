import os
import sys
import inspect
import importlib.metadata

project = "excee"
copyright = "2026, Zachary J Weiner"
author = "Zachary J Weiner"

release = importlib.metadata.version("excee")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "description"
typehints_defaults = "comma"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

import subprocess
try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
except Exception:
    commit_hash = "main"

linkcode_url = f"https://github.com/zachjweiner/{project}/blob/{commit_hash}/excee/{{filepath}}{{linespec}}"


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    submod = sys.modules.get(info["module"])
    if submod is None:
        return None

    obj = submod
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        obj = inspect.unwrap(obj)
        fn = inspect.getsourcefile(obj)
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    except OSError:
        linespec = ""

    import excee
    filepath = os.path.relpath(fn, start=os.path.dirname(excee.__file__))
    filepath = filepath.replace(os.path.sep, "/")
    return linkcode_url.format(filepath=filepath, linespec=linespec)


def fix_numpy_typing_aliases(app, env, node, contnode):
    target = node.get("reftarget", "")

    if target.startswith("numpy._typing") or target.startswith("matplotlib.typing"):
        node["reftarget"] = target.replace("numpy._typing", "numpy.typing")
        node["reftype"] = "obj"
        return None


def setup(app):
    app.connect("missing-reference", fix_numpy_typing_aliases, priority=400)
