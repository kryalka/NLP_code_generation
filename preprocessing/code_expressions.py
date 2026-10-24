import ast
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Type:
    kind: str
    items: tuple = ()


NUMBER = Type('number')
BOOL = Type('bool')
STRING = Type('string')
NONE = Type('none')
UNKNOWN = Type('unknown')
EMPTY = Type('empty')

BUILTINS = {
    'int', 'float', 'round', 'abs', 'pow', 'str', 'repr', 'format',
    'bool', 'all', 'any', 'isinstance', 'len', 'sum', 'min', 'max',
    'range', 'enumerate', 'zip', 'list', 'tuple', 'set', 'sorted',
    'reversed', 'divmod', 'print',
}

@dataclass(eq=False)
class Function:
    node: object
    closure: dict


def merge(*types):
    """
    Объединить типы, например после разных ветвей if
    """
    types = [value_type for value_type in types if value_type != EMPTY]
    if not types:
        return EMPTY
    first = types[0]
    if all(value_type == first for value_type in types):
        return first
    if UNKNOWN in types:
        return UNKNOWN
    if first.kind not in ('list', 'set', 'iter'):
        return Type('mixed')
    for value_type in types:
        if value_type.kind != first.kind:
            return Type('mixed')
    items = [element(value_type) for value_type in types]
    return Type(first.kind, (merge(*items),))


def element(value_type):
    """
    Тип одного элемента списка, кортежа, словаря или строки
    """
    if value_type.kind in ('list', 'set', 'iter'):
        return value_type.items[0]
    if value_type.kind == 'tuple':
        return merge(*value_type.items)
    if value_type.kind == 'dict':
        return value_type.items[0]
    if value_type == STRING:
        return STRING
    return UNKNOWN


def expression(checker, node, variables):
    """
    Определить тип выражения, не выполняя его
    """
    if node is None:
        return NONE

    # Значения, записанные прямо в коде: 5, 2.5, True, "text", None
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                return Type('nonfinite')
            return NUMBER
        if type(node.value) is bool:
            return BOOL
        if isinstance(node.value, str):
            return STRING
        if node.value is None:
            return NONE
        return Type('non_numeric')

    # Переменная или имя функции, например result, sum или helper
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id == '__name__':
            return STRING
        if node.id in BUILTINS:
            return Type('builtin', (node.id,))
        return checker.unknown()

    # Операции с одним значением: -x, +x, not x
    if isinstance(node, ast.UnaryOp):
        value_type = expression(checker, node.operand, variables)
        if isinstance(node.op, ast.Not):
            return BOOL
        if value_type in (NUMBER, BOOL):
            return NUMBER
        return checker.unknown()

    # Операции с двумя значениями: x + y, x * y, "text" + name
    if isinstance(node, ast.BinOp):
        left = expression(checker, node.left, variables)
        right = expression(checker, node.right, variables)
        if left in (NUMBER, BOOL) and right in (NUMBER, BOOL):
            return NUMBER
        if isinstance(node.op, ast.Add) and left == right == STRING:
            return STRING
        if isinstance(node.op, ast.Mod) and left == STRING:
            return STRING
        if isinstance(node.op, ast.Mult) and STRING in (left, right) and NUMBER in (left, right):
            return STRING
        if isinstance(node.op, (ast.Add, ast.Mult)) and left.kind in ('list', 'tuple'):
            if right == NUMBER:
                return left
            items = merge(element(left), element(right))
            return Type(left.kind, (items,))
        return checker.unknown()

    # Сравнения возвращают bool, а and/or могут возвращать сами значения
    if isinstance(node, ast.Compare):
        expression(checker, node.left, variables)
        for value in node.comparators:
            expression(checker, value, variables)
        return BOOL
    if isinstance(node, ast.BoolOp):
        types = [expression(checker, value, variables) for value in node.values]
        return merge(*types)

    # Учитываем оба варианта в выражении x if condition else y
    if isinstance(node, ast.IfExp):
        expression(checker, node.test, variables)
        if_true = expression(checker, node.body, variables)
        if_false = expression(checker, node.orelse, variables)
        return merge(if_true, if_false)

    # f"Answer: {result}" возвращает строку
    if isinstance(node, ast.JoinedStr):
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                expression(checker, value.value, variables)
        return STRING

    # запоминаем типы их элементов
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = tuple(expression(checker, value, variables) for value in node.elts)
        if isinstance(node, ast.Tuple):
            return Type('tuple', items)
        kind = 'set'
        if isinstance(node, ast.List):
            kind = 'list'
        return Type(kind, (merge(*items),))
    if isinstance(node, ast.Dict):
        keys = [expression(checker, key, variables) for key in node.keys]
        values = [expression(checker, value, variables) for value in node.values]
        return Type('dict', (merge(*keys), merge(*values)))

    # [x * x for x in range(5)] и похожие конструкции
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        local = dict(variables)
        for generator in node.generators:
            sequence = expression(checker, generator.iter, local)
            checker.assign(generator.target, element(sequence), local)
            for condition in generator.ifs:
                expression(checker, condition, local)
        kind = 'iter'
        if isinstance(node, ast.ListComp):
            kind = 'list'
        elif isinstance(node, ast.SetComp):
            kind = 'set'
        item = expression(checker, node.elt, local)
        return Type(kind, (item,))

    # Получение элемента или среза: values[0], values[:2]
    if isinstance(node, ast.Subscript):
        value = expression(checker, node.value, variables)
        if isinstance(node.slice, ast.Slice):
            return value
        if value.kind == 'tuple' and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
            try:
                return value.items[node.slice.value]
            except IndexError:
                return checker.unknown()
        if value.kind == 'dict':
            return value.items[1]
        return element(value)

    # Значения и функции из библиотек, например math.pi и math.sqrt
    if isinstance(node, ast.Attribute):
        base = expression(checker, node.value, variables)
        if base.kind == 'module':
            path = base.items[0] + '.' + node.attr
            if path in {'math.pi', 'math.e', 'math.tau'}:
                return NUMBER
            return Type('library', (path,))
        return checker.unknown()

    if isinstance(node, ast.Call):
        return check_call(checker, node, variables)
    return checker.unknown()


def check_call(checker, node, variables):
    """Тип результата sum(...), math.sqrt(...), helper(...) и других вызовов"""
    args = [expression(checker, argument, variables) for argument in node.args]

    if isinstance(node.func, ast.Attribute):
        base = expression(checker, node.func.value, variables)
        method = node.func.attr
        # Словарь: data.keys(), data.values(), data.items()
        if base.kind == 'dict' and method in {'keys', 'values', 'items'}:
            if method == 'keys':
                item = base.items[0]
            elif method == 'values':
                item = base.items[1]
            else:
                item = Type('tuple', base.items)
            return Type('iter', (item,))

        # Изменение коллекции: values.append(x), values.extend(items), values.add(x)
        if base.kind in {'list', 'set'} and method in {'append', 'add', 'extend'}:
            if isinstance(node.func.value, ast.Name):
                item = args[0]
                if method == 'extend':
                    item = element(item)
                old = variables.get(node.func.value.id)
                if id(old) in checker.external_containers:
                    checker.uncertain = True
                items = merge(element(base), item)
                new = Type(base.kind, (items,))
                # Если x и y ссылаются на один список, обновляем тип обоих
                for name in list(variables):
                    if variables[name] is old:
                        variables[name] = new
                return NONE

    # Вызов функции, определённой в самой программе: helper(x)
    function_type = expression(checker, node.func, variables)
    if function_type.kind == 'function':
        return checker.function(function_type.items[0], args, node.keywords, variables)
    name = ''
    if function_type.kind in {'builtin', 'library'}:
        name = function_type.items[0]
    for keyword in node.keywords:
        expression(checker, keyword.value, variables)

    if function_type.kind == 'builtin':
        # float('nan') и float('inf') не подходят под числовой ответ
        if name == 'float' and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                text = first_arg.value.strip().lower().lstrip('+-')
                if text in {'inf', 'infinity', 'nan'}:
                    return Type('nonfinite')

        # Преобразования и числовые функции
        if name in {'int', 'float', 'round', 'abs', 'pow'} and args:
            if args[0].kind in {'number', 'bool', 'string'}:
                return NUMBER
            return checker.unknown()

        # Строковые и логические результаты
        if name in {'str', 'repr', 'format'}:
            return STRING
        if name in {'bool', 'all', 'any', 'isinstance'}:
            return BOOL

        # len, sum, min и max
        if name == 'len' and args and args[0].kind in {'list', 'tuple', 'set', 'dict', 'string', 'iter'}:
            return NUMBER
        if name in {'sum', 'min', 'max'} and args:
            if len(args) == 1 or name == 'sum':
                item = element(args[0])
            else:
                item = merge(*args)
            if item in (NUMBER, BOOL, EMPTY):
                return NUMBER
            if name != 'sum':
                return item
            return checker.unknown()

        # Последовательности для циклов и вычислений
        if name == 'range':
            return Type('iter', (NUMBER,))
        if name == 'enumerate' and args:
            pair = Type('tuple', (NUMBER, element(args[0])))
            return Type('iter', (pair,))
        if name == 'zip':
            items = tuple(element(value_type) for value_type in args)
            pair = Type('tuple', items)
            return Type('iter', (pair,))
        if name in {'list', 'tuple', 'set', 'sorted', 'reversed'} and args:
            kind = name
            if name in {'list', 'sorted'}:
                kind = 'list'
            elif name == 'reversed':
                kind = 'iter'
            return Type(kind, (element(args[0]),))
        if name == 'divmod':
            return Type('tuple', (NUMBER, NUMBER))

        # Оставшийся вызов print не позволяет подтвердить программу
        if name == 'print':
            checker.uncertain = True
            return NONE

    # Поддерживаемые функции из math, fractions, decimal и statistics
    if function_type.kind == 'library':
        if name.startswith('math.'):
            method = name[5:]
            if hasattr(math, method):
                if method in {'isfinite', 'isinf', 'isnan', 'isclose'}:
                    return BOOL
                if method in {'frexp', 'modf'}:
                    return Type('tuple', (NUMBER, NUMBER))
                if callable(getattr(math, method)):
                    return NUMBER
        if name in {'fractions.Fraction', 'decimal.Decimal'}:
            return NUMBER
        if name in {'statistics.mean', 'statistics.median', 'statistics.fmean', 'statistics.pstdev', 'statistics.stdev'}:
            if args and element(args[0]) in (NUMBER, BOOL):
                return NUMBER
            return checker.unknown()
    return checker.unknown()
