# EMBRS: Engineering Model for Burning and Real-time Suppression

EMBRS is a real-time fire simulation software that provides users a sandbox for testing fire
suppression algorithms and strategies in a Python framework.

## Installation
EMBRS can be installed by downloading source code or via the PyPI package manager with `pip`.

The simplest method is using `pip` with the following command:

```bash
pip install embrs
```

Developers who would like to inspect the source code can install EMBRS by downloading the git repository from GitHub and use `pip` to install it locally. The following terminal commands can be used to do this:

```bash
# Download source code from the 'main' branch
git clone -b main https://github.com/AREAL-GT/embrs.git

# Install EMBRS
pip install -e embrs

```

## Usage
### Launching EMBRS Applications
With the package installed in your environment, launch the GUIs directly via the modules:

```bash
# Run a simulation (GUI if no args; --config to load a .cfg)
python -m main.py
python -m main.py --config path/to/config.cfg

# Visualize a finished run
python -m visualization_tool.py

# Create an EMBRS map
python -m map_generator.py
```

Each command opens a GUI for that workflow. Read the rest of this site for the inputs each tool expects.
