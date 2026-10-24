import ast
import io
import re
import tokenize

from .code_returns import check_return


PURE_BUILTINS = frozenset({'str', 'repr', 'format', 'int', 'float', 'round', 'abs', 'len'})

# print(5), print(result), print(x + 2), print([x, y])
SIMPLE_NODES = (
    ast.Constant, ast.Name, ast.UnaryOp, ast.BinOp,
    ast.BoolOp, ast.Compare, ast.IfExp, ast.List, ast.Tuple,
)

# print(f"Answer: {result}")
FORMATTED_NODES = (ast.JoinedStr, ast.FormattedValue)

def diff(new, old):
    return new if new != old else None


def process_code(source, has_hash):
    # Если source не строка или пустая/из пробелов, то парсить нечего
    if not isinstance(source, str) or not source.strip():
        return None, None, False

    # Убираем комментарии только там, где есть символ #
    # поэтому надо отобрать только те сторки, где есть символ комментария
    no_comments = remove_comments(source) if has_hash else source

    try:
        # парсим текст в ast
        tree = ast.parse(no_comments)
        # берем код без комментариев
        cleaned = no_comments
        # отбираем строки где есть/нет print
        if 'print' in no_comments:
            cleaned, tree, _ = remove_prints(no_comments, tree)
        cleaned = remove_empty_lines(cleaned, tree)
        # проверяем корректный тип ответа
        is_numeric = check_return(tree, 'yield' in cleaned)
        return diff(no_comments, source), diff(cleaned, no_comments), is_numeric

    except (SyntaxError, ValueError, TypeError, RecursionError, IndexError):
        # программы, которые не удалось проверить, исключаем из датасета (только у train)
        return diff(no_comments, source), None, False


def process_chunk(task):
    codes, flags = task
    return [process_code(source, flag) for source, flag in zip(codes, flags)]


def remove_empty_lines(source, tree):
    lines = io.StringIO(source, newline='').readlines()
    protected = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.JoinedStr)):
            protected.update(range(node.lineno, node.end_lineno + 1))

    result = []
    for number, line in enumerate(lines, start=1):
        # После явного переноса пустая строка может завершать выражение
        continued = number > 1 and lines[number - 2].rstrip('\r\n').endswith('\\')
        if line.strip(' \t\r\n') or number in protected or continued:
            result.append(line)
    return ''.join(result)


def remove_comments(source):
    comments = []
    try:
        reader = io.StringIO(source, newline=None).readline
        for token in tokenize.generate_tokens(reader):
            if token.type == tokenize.COMMENT:
                comments.append(token)
    except (tokenize.TokenError, SyntaxError):
        return source

    if not comments:
        return source

    line_starts = [0] + [match.end() for match in re.finditer(r'\r\n?|\n', source)]
    parts = []
    previous_end = 0
    for token in comments:
        row = token.start[0] - 1
        start = line_starts[row] + token.start[1]
        end = line_starts[row] + token.end[1]
        parts.append(source[previous_end:start])
        previous_end = end
    parts.append(source[previous_end:])
    return ''.join(parts)


def pure_expression(node):
    """
    Проверяем, можно ли вычислить выражение, ничего не сломав
    константы и переменные — да, арифметика, сравнения, списки, f-строки — да, но надо проверить части выражений
    len, str, int и тд - да
    остальное - нет
    """
    # проверяем: константа, имя, арифметика, сравнение, список, f-строка, то всё ок
    # проверяем части выражения, чтобы отследить такое: [1, "2", 3]
    if isinstance(node, SIMPLE_NODES + FORMATTED_NODES):
        return all(
            pure_expression(child)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        )

    if not isinstance(node, ast.Call):
        return False

    func = node.func
    # пример: print(...), len(x), str(y), round(z, 2)
    if isinstance(func, ast.Name):
        if func.id not in PURE_BUILTINS:
            return False
    elif isinstance(func, ast.Attribute):
        # тут мы разрешаем только одно: .format() у строкового литерала
        if not isinstance(func.value, ast.Constant):
            return False
        if not isinstance(func.value.value, str) or func.attr != 'format':
            return False
    else:
        return False

    return (
        all(pure_expression(argument) for argument in node.args)
        and all(keyword.arg is not None and pure_expression(keyword.value)
                for keyword in node.keywords)
    )


def removable_print(node):
    """
    Удаляем только отдельный вызов print с допустимыми аргументами
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    # самостоятельный print()
    if not isinstance(call.func, ast.Name) or call.func.id != 'print':
        return False
    # print(), print(result), print(x, y) и другие разрешённые аргументы
    for argument in call.args:
        if not pure_expression(argument):
            return False
    for keyword in call.keywords:
        if keyword.arg not in {'sep', 'end', 'flush'}:
            return False
        if not pure_expression(keyword.value):
            return False
    return True


class DropPrints(ast.NodeTransformer):
    """
    Удаляем выбранные print из AST и добавляем pass в пустые блоки
    """
    def __init__(self, removable):
        self.removable = removable

    def visit_Expr(self, node):
        if node in self.removable:
            return None
        return self.generic_visit(node)

    def generic_visit(self, node):
        nonempty = [field for field in ('body', 'orelse', 'finalbody') if getattr(node, field, None)]
        node = super().generic_visit(node)
        for field in nonempty:
            if not getattr(node, field):
                # если print в if, for или функции, то не должно быть пусто после удаления
                setattr(node, field, [ast.Pass()])
        return node


def remove_prints(source, tree):
    """
    Находим print, делаем замену пустых блоков
    """
    nodes = [node for node in ast.walk(tree) if removable_print(node)]
    if not nodes:
        return source, tree, 0

    removable = set(nodes)

    pass_nodes = set()
    for parent in ast.walk(tree):
        for field in ('body', 'orelse', 'finalbody'):
            block = getattr(parent, field, None)
            if isinstance(block, list) and block:
                if all(node in removable for node in block):
                    pass_nodes.add(block[0])

    lines = source.split('\n')
    can_remove_lines = '\r' not in source and '\f' not in source
    edits = []
    for node in nodes:
        if not can_remove_lines:
            break
        first = node.lineno - 1
        last = node.end_lineno - 1
        before = lines[first].encode('utf-8')[:node.col_offset].decode('utf-8')
        after = lines[last].encode('utf-8')[node.end_col_offset:].decode('utf-8')
        if before.strip() or after.strip():
            can_remove_lines = False

        replacement = ''
        if node in pass_nodes:
            replacement = before + 'pass'
        edits.append((first, last, replacement))

    cleaned_tree = DropPrints(removable).visit(tree)
    cleaned_tree = ast.fix_missing_locations(cleaned_tree)
    if can_remove_lines:
        for first, last, replacement in sorted(edits, reverse=True):
            lines[first:last + 1] = [replacement]
        cleaned = '\n'.join(lines)
    else:
        cleaned = ast.unparse(cleaned_tree) + '\n'
    return cleaned, ast.parse(cleaned), len(nodes)
