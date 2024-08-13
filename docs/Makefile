# Minimal Makefile for Sphinx documentation
# You can set these variables from the command line.
SPHINXOPTS    = -vv --keep-going
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = AutoMLQuantILDetect
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. Check Sphinx's command line parsing for the detail
# of what is going on here.
%: Makefile
	@echo "Building documentation with verbosity level: $(SPHINXOPTS)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)