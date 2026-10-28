import ast,operator,re

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Pow: operator.pow, ast.Div: operator.truediv}
CMP = {ast.Gt: operator.gt, ast.Lt: operator.lt, ast.Eq: operator.eq, ast.GtE: operator.ge, ast.LtE: operator.le}

def expr(n, env):
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        return n.value
    if isinstance(n, ast.Name):
        return env[n.id]
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        left, right = (expr(n.left, env), expr(n.right, env))
        if isinstance(n.op, ast.Pow) and abs(right) > 8:
            raise ValueError('Esponente fuori limite')
        return OPS[type(n.op)](left, right)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -expr(n.operand, env)
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and (type(n.ops[0]) in CMP):
        return CMP[type(n.ops[0])](expr(n.left, env), expr(n.comparators[0], env))
    if isinstance(n, ast.IfExp):
        return expr(n.body if expr(n.test, env) else n.orelse, env)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == 'max') and (not n.keywords):
        return max((expr(a, env) for a in n.args))
    raise ValueError('Costrutto non supportato dall’interprete di test')

def verify_code(text, name, tests):
    blocks = re.findall('```(?:python)?\\s*\\n(.*?)```', text, re.S)
    code = blocks[0] if blocks else text.strip()
    try:
        tree = ast.parse(code)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError('Serve una sola funzione completa')
        fn = tree.body[0]
        if fn.name != name or fn.decorator_list or len(fn.body) != 1 or (not isinstance(fn.body[0], ast.Return)):
            raise ValueError('Funzione assente o corpo non supportato')
        for inputs, expected in tests:
            if len(fn.args.args) != len(inputs):
                raise ValueError('Argomenti errati')
            actual = expr(fn.body[0].value, dict(zip([a.arg for a in fn.args.args], inputs)))
            if actual != expected:
                return (False, {'inputs': inputs, 'expected': expected, 'actual': actual})
        return (True, {'cases_passed': len(tests), 'method': 'interprete AST limitato; nessun exec'})
    except Exception as e:
        return (False, {'error': str(e), 'manual_review_if_unsupported': True})
