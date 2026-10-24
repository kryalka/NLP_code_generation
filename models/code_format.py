HEADER = "def solve() -> int:"
FOOTER = "    return result"


def strip_solve_wrapper(code):
    lines = code.splitlines()
    if len(lines) >= 2 and lines[0] == HEADER and lines[-1] == FOOTER:
        body = "\n".join(lines[1:-1])
        return body + ("\n" if body else "")
    return code


def restore_solve_wrapper(body):
    body = body.rstrip("\r\n")
    lines = body.splitlines()
    if len(lines) >= 2 and lines[0] == HEADER and lines[-1] == FOOTER:
        return body + "\n"
    return f"{HEADER}\n{body}\n{FOOTER}\n"
