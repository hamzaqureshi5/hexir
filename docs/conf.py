# Sphinx configuration for the Hexir documentation.
#
# Pages are written in Markdown (MyST) rather than reStructuredText so that
# they read the same on GitHub and on the docs site.

project = "Hexir"
copyright = "2026, Hexir contributors"
author = "Hexir contributors"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# Markdown niceties: ::: fenced directives, and $math$.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# Turn ```mermaid fences into the mermaid directive, so they render as diagrams
# rather than as code blocks. Written as fences so the same source also renders
# on GitHub, which understands mermaid natively.
myst_fence_as_directive = ["mermaid"]

# Pygments has no MLIR lexer. The ```mlir fences are kept because they are
# correct and GitHub highlights them; here they just render unhighlighted.
suppress_warnings = ["misc.highlighting_failure"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Hexir"
html_theme_options = {
    "source_repository": "https://github.com/hamzaqureshi5/hexir",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Diagrams render client-side, so they work on GitHub Pages with no build-time
# browser or image generation.
mermaid_output_format = "raw"
