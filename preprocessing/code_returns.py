import ast

from .code_expressions import (
    Type, Function, NUMBER, BOOL, NONE, UNKNOWN, merge, element, expression,
)


class ReturnChecker:
    def __init__(self, check_generators=True):
        self.uncertain = False  # Есть ли конструкции, тип которых не удалось определить
        self.active = set()  # Функции, которые уже разбираем, чтобы не уйти в рекурсию
        self.external_containers = set()  # Списки, словари и множества из внешней функции
        self.check_generators = check_generators

    def unknown(self):
        self.uncertain = True
        return UNKNOWN

    def assign(self, target, value, variables):
        if isinstance(target, ast.Name):
            variables[target.id] = value
            return

        # Распаковка: x, y = (2, 3)
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, part in enumerate(target.elts):
                if value.kind == 'tuple' and index < len(value.items):
                    item = value.items[index]
                else:
                    item = element(value)
                self.assign(part, item, variables)
            return

        # Присваивание элементу списка: values[0] = 5
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            old = variables.get(target.value.id, UNKNOWN)
            if old.kind == 'list':
                if id(old) in self.external_containers:
                    self.uncertain = True
                item = merge(element(old), value)
                new = Type('list', (item,))
                # Учитываем и другие имена этого же списка: other = values
                for name in list(variables):
                    if variables[name] is old:
                        variables[name] = new
                return

        self.uncertain = True

    def block(self, statements, variables):
        returns = []
        can_continue = True
        for node in statements:
            if not can_continue:
                break

            # Запоминаем определения функций и доступные им переменные
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.decorator_list:
                    self.uncertain = True
                defaults = list(node.args.defaults)
                for value in node.args.kw_defaults:
                    if value is not None:
                        defaults.append(value)
                for default in defaults:
                    expression(self, default, variables)
                function = Function(node, variables)
                variables[node.name] = Type('function', (function,))

            # Запоминаем имена библиотек и импортированных функций
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    if isinstance(node, ast.ImportFrom):
                        path = (node.module or '') + '.' + alias.name
                        variables[name] = Type('library', (path,))
                    else:
                        variables[name] = Type('module', (alias.name,))

            # Присваивания: result = 5, x, y = (2, 3)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = expression(self, node.value, variables)
                if isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    targets = [node.target]
                for target in targets:
                    self.assign(target, value, variables)

            # Изменения вроде result += 2
            elif isinstance(node, ast.AugAssign):
                value = expression(self, node.value, variables)
                old = expression(self, node.target, variables)
                if old in (NUMBER, BOOL) and value in (NUMBER, BOOL):
                    result = NUMBER
                else:
                    result = self.unknown()
                self.assign(node.target, result, variables)

            # После return остальные команды этого блока не выполняются
            elif isinstance(node, ast.Return):
                returns.append(expression(self, node.value, variables))
                can_continue = False

            # Объединяем результаты возможных ветвей if/else
            elif isinstance(node, ast.If):
                expression(self, node.test, variables)
                bodies = (node.body, node.orelse)
                if isinstance(node.test, ast.Constant):
                    if node.test.value:
                        bodies = (node.body,)
                    else:
                        bodies = (node.orelse,)
                branches = [self.block(body, dict(variables)) for body in bodies]
                continuing = []
                for state, values, branch_continues in branches:
                    returns.extend(values)
                    if branch_continues:
                        continuing.append(state)
                if continuing:
                    names = set()
                    for state in continuing:
                        names.update(state)
                    for name in names:
                        types = [state.get(name, UNKNOWN) for state in continuing]
                        variables[name] = merge(*types)
                can_continue = bool(continuing)

            # Цикл может не выполниться, поэтому учитываем типы до и после его тела
            elif isinstance(node, (ast.For, ast.While)):
                before = dict(variables)
                if isinstance(node, ast.For):
                    sequence = expression(self, node.iter, variables)
                    self.assign(node.target, element(sequence), variables)
                else:
                    expression(self, node.test, variables)
                body, values, _ = self.block(node.body, dict(variables))
                returns.extend(values)
                names = set(before)
                names.update(body)
                for name in names:
                    before_type = before.get(name, UNKNOWN)
                    after_type = body.get(name, UNKNOWN)
                    variables[name] = merge(before_type, after_type)
                if node.orelse:
                    variables, values, _ = self.block(node.orelse, variables)
                    returns.extend(values)

            elif isinstance(node, ast.Expr):
                expression(self, node.value, variables)
            elif isinstance(node, (ast.Break, ast.Continue)):
                can_continue = False
            elif not isinstance(node, ast.Pass):
                self.uncertain = True
        return variables, returns, can_continue

    def function(self, function, args, keywords=(), caller=None):
        node = function.node
        if id(node) in self.active or isinstance(node, ast.AsyncFunctionDef) or node.decorator_list:
            return self.unknown()
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            return self.unknown()

        # Сопоставляем параметры с аргументами
        parameters = list(node.args.posonlyargs) + list(node.args.args)
        local = dict(function.closure)
        required_count = len(parameters) - len(node.args.defaults)
        defaults = [None] * required_count + node.args.defaults
        assigned = {}
        for parameter, value_type in zip(parameters, args):
            assigned[parameter.arg] = value_type
        parameter_names = {parameter.arg for parameter in parameters}
        for keyword in keywords:
            if keyword.arg is None:
                return self.unknown()
            if keyword.arg in assigned or keyword.arg not in parameter_names:
                return self.unknown()
            assigned[keyword.arg] = expression(self, keyword.value, caller or local)
        if len(args) > len(parameters):
            return self.unknown()
        for parameter, default in zip(parameters, defaults):
            if parameter.arg in assigned:
                local[parameter.arg] = assigned[parameter.arg]
            elif default is not None:
                local[parameter.arg] = expression(self, default, function.closure)
            else:
                return self.unknown()

        # Функция с yield возвращает генератор
        stack = []
        if self.check_generators:
            stack = list(node.body)
        while stack:
            part = stack.pop()
            if isinstance(part, (ast.Yield, ast.YieldFrom)):
                return Type('generator')
            if not isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.extend(ast.iter_child_nodes(part))
        self.active.add(id(node))

        # Изменение переданного списка может повлиять на вызывающую функцию
        previous_external = self.external_containers
        self.external_containers = set(previous_external)
        for value_type in list(args) + list(function.closure.values()):
            if value_type.kind in {'list', 'dict', 'set'}:
                self.external_containers.add(id(value_type))
        _, returns, can_continue = self.block(node.body, local)
        self.external_containers = previous_external
        self.active.remove(id(node))

        # Если дошли до конца функции без return, Python возвращает None
        if can_continue:
            returns.append(NONE)
        return merge(*returns)


def check_return(tree, check_generators=True):
    checker = ReturnChecker(check_generators)
    variables, _, _ = checker.block(tree.body, {})
    solve = variables.get('solve', UNKNOWN)
    if solve.kind != 'function':
        return False
    if isinstance(solve.items[0].node, ast.AsyncFunctionDef):
        return False

    result = checker.function(solve.items[0], [])
    return result == NUMBER and not checker.uncertain
