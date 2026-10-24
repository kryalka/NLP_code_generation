import json
import subprocess
import sys


WORKER = '''
import contextlib
from decimal import Decimal
import json
import math
import numbers
import sys

code = sys.stdin.read()
namespace = {"__name__": "__generated__"}
with contextlib.redirect_stdout(sys.stderr):
    exec(code, namespace)
    value = namespace["solve"]()

numeric = isinstance(value, (numbers.Real, Decimal)) and not isinstance(value, bool)
if not numeric and "sympy" in sys.modules:
    numeric = isinstance(value, sys.modules["sympy"].Number) and value.is_real is True
if not numeric:
    raise ValueError("Результат не является числом")
value = float(value)
if not math.isfinite(value):
    raise ValueError("Результат не является конечным числом")
print(json.dumps(value))
'''


def run_code(code):
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER], input=code, text=True, encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2, check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError):
        return None
