# Development Setup

Target runtime is Python 3.11. Create a clean environment and install the editable
package with development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

The current smoke evidence was executed in the provided workspace interpreter,
Python 3.14.3. CI and future Docker work must use Python 3.11 as specified by the
project blueprint.

