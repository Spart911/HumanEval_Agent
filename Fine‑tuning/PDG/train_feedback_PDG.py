import os
import ast
import torch
import networkx as nx
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import gc

# Отключаем предупреждение parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- 1. Параметры ----------------
LOCAL_MODEL_DIR = "/home/nyuroprint/Jupyter/Qwen2.5-Coder-3B"  # Папка с уже скачанной моделью
MAX_LEN = 512
EPOCHS = 2
BATCH_SIZE = 1
LR = 5e-6
OUTPUT_DIR = "./pdg_feedback_trained_github"
LIMIT = 10000

REPOS = [
    "https://github.com/psf/requests.git",
    "https://github.com/pallets/flask.git",
    "https://github.com/pandas-dev/pandas.git",
    "https://github.com/numpy/numpy.git",
    "https://github.com/scipy/scipy.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    "https://github.com/matplotlib/matplotlib.git",
    "https://github.com/plotly/plotly.py.git",
    "https://github.com/pytorch/pytorch.git",
    "https://github.com/tensorflow/tensorflow.git"
]

CLONE_DIR = "./github_repos"
os.makedirs(CLONE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Проверяем, что модель существует локально
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"❌ Локальная модель не найдена в папке: {LOCAL_MODEL_DIR}")
    print("Убедитесь, что модель скачана в эту папку")
    exit(1)
else:
    print(f"✅ Локальная модель найдена в: {LOCAL_MODEL_DIR}")

# ---------------- 2. Клонируем репозитории ----------------
print("📁 Клонируем репозитории...")
for repo in REPOS:
    repo_name = repo.split("/")[-1].replace(".git", "")
    dest = os.path.join(CLONE_DIR, repo_name)
    if not os.path.exists(dest):
        print(f"Cloning {repo} …")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo, dest],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"⚠️ Проблема с клонированием {repo}: {result.stderr}")
            else:
                print(f"✅ Успешно клонирован {repo}")
        except Exception as e:
            print(f"❌ Ошибка клонирования {repo}: {e}")

# ---------------- 3. Построение PDG (Program Dependence Graph) ----------------
# PDG = AST nodes + control-flow edges (CFG-like) + data-flow edges (DFG-like)

def build_pdg_from_ast(tree: ast.AST, graph: nx.DiGraph):
    """
    Строит PDG (Program Dependence Graph) из AST.
    graph -- networkx.DiGraph() — узлы: label, type, name(opt)
    Алгоритм:
      - добавляем AST-узлы (FunctionDef, If, For, Assign, Name, Constant, Call ...)
      - создаём control-edges между последовательными операторами ("next"), из условий в тела ("control")
      - создаём data-flow edges: def -> use (на основе простейшей локальной карты определений)
    Замечание: это упрощённая и прагматичная реализация PDG, пригодная для извлечения последовательностей/фич.
    """
    if "next_id" not in graph.graph:
        graph.graph["next_id"] = 0

    def new_id():
        nid = graph.graph["next_id"]
        graph.graph["next_id"] += 1
        return nid

    # будем хранить карту последних определений переменных на уровне текущей области
    def walk_block(statements, local_defs, prev_stmt_node=None):
        """Обрабатывает список stmt, создаёт control 'next' edges между ними."""
        last_node = prev_stmt_node
        for stmt in statements:
            node_id = walk(stmt, local_defs)
            if node_id is None:
                continue
            # соединяем контрольным ребром предыдущий оператор -> текущий (последовательность)
            if last_node is not None:
                graph.add_edge(last_node, node_id, label="next", type="control")
            last_node = node_id
        return last_node

    def walk(n, local_defs):
        """Рекурсивная обработка AST-узла. Возвращает node_id(или список id для выражений)."""
        if n is None:
            return None

        # ---- Leaf nodes ----
        if isinstance(n, ast.Name):
            nid = new_id()
            ctx = "Load" if isinstance(n.ctx, ast.Load) else "Store" if isinstance(n.ctx, ast.Store) else type(n.ctx).__name__
            graph.add_node(nid, label=f"Name:{n.id} ({ctx})", type="Name", name=n.id, ctx=ctx)
            # если загрузка — добавляем ребро от последнего def->use
            if isinstance(n.ctx, ast.Load) and n.id in local_defs:
                graph.add_edge(local_defs[n.id], nid, label="def->use", type="data")
            return nid

        if isinstance(n, ast.Constant):
            nid = new_id()
            graph.add_node(nid, label=f"Const:{repr(n.value)}", type="Const")
            return nid

        # ---- Expressions ----
        if isinstance(n, ast.BinOp):
            left_id = walk(n.left, local_defs)
            right_id = walk(n.right, local_defs)
            op_id = new_id()
            op_name = type(n.op).__name__
            graph.add_node(op_id, label=f"BinOp:{op_name}", type="BinOp")
            for lid in (left_id, right_id):
                if lid is not None:
                    graph.add_edge(lid, op_id, label="operand", type="data")
            return op_id

        if isinstance(n, ast.UnaryOp):
            operand_id = walk(n.operand, local_defs)
            op_id = new_id()
            op_name = type(n.op).__name__
            graph.add_node(op_id, label=f"UnaryOp:{op_name}", type="UnaryOp")
            if operand_id is not None:
                graph.add_edge(operand_id, op_id, label="operand", type="data")
            return op_id

        if isinstance(n, ast.Call):
            func_id = walk(n.func, local_defs)
            call_id = new_id()
            graph.add_node(call_id, label="Call", type="Call")
            if func_id is not None:
                graph.add_edge(func_id, call_id, label="callfunc", type="data")
            for arg in n.args:
                aid = walk(arg, local_defs)
                if aid is not None:
                    graph.add_edge(aid, call_id, label="arg", type="data")
            for kw in getattr(n, "keywords", []):
                v = walk(kw.value, local_defs)
                if v is not None:
                    graph.add_edge(v, call_id, label=f"kw:{kw.arg}", type="data")
            return call_id

        # ---- Statements ----
        if isinstance(n, ast.Assign):
            # RHS value
            value_id = walk(n.value, local_defs)
            assign_id = new_id()
            graph.add_node(assign_id, label="Assign", type="Assign")
            if value_id is not None:
                graph.add_edge(value_id, assign_id, label="value", type="data")
            # targets
            for t in n.targets:
                if isinstance(t, ast.Name):
                    def_id = new_id()
                    graph.add_node(def_id, label=f"Def:{t.id}", type="Def", name=t.id)
                    graph.add_edge(assign_id, def_id, label="assign->def", type="data")
                    # обновляем локальную карту определений (последнее определение)
                    local_defs = dict(local_defs)
                    local_defs[t.id] = def_id
                else:
                    # рекурсивно обрабатываем сложные targets (tuple, subscript, attr)
                    target_out = walk(t, local_defs)
                    if target_out is not None:
                        graph.add_edge(assign_id, target_out, label="assign->target", type="data")
            return assign_id

        if isinstance(n, ast.AugAssign):
            target_id = walk(n.target, local_defs)
            value_id = walk(n.value, local_defs)
            op_id = new_id()
            op_name = type(n.op).__name__
            graph.add_node(op_id, label=f"AugOp:{op_name}", type="AugOp")
            for src in (target_id, value_id):
                if src is not None:
                    graph.add_edge(src, op_id, label="operand", type="data")
            if isinstance(n.target, ast.Name):
                def_id = new_id()
                graph.add_node(def_id, label=f"Def:{n.target.id}", type="Def", name=n.target.id)
                graph.add_edge(op_id, def_id, label="result", type="data")
                local_defs = dict(local_defs)
                local_defs[n.target.id] = def_id
                return def_id
            return op_id

        if isinstance(n, ast.Return):
            val_id = walk(n.value, local_defs) if n.value else None
            ret_id = new_id()
            graph.add_node(ret_id, label="Return", type="Return")
            if val_id is not None:
                graph.add_edge(val_id, ret_id, label="ret", type="data")
            return ret_id

        if isinstance(n, ast.Expr):
            return walk(n.value, local_defs)

        if isinstance(n, ast.Compare):
            left_id = walk(n.left, local_defs)
            comp_id = new_id()
            graph.add_node(comp_id, label="Compare", type="Compare")
            if left_id is not None:
                graph.add_edge(left_id, comp_id, label="left", type="data")
            for op, comp in zip(n.ops, n.comparators):
                cid = walk(comp, local_defs)
                if cid is not None:
                    graph.add_edge(cid, comp_id, label="comp", type="data")
            return comp_id

        if isinstance(n, ast.If):
            # Условие
            test_id = walk(n.test, local_defs)
            if_id = new_id()
            graph.add_node(if_id, label="If", type="If")
            if test_id is not None:
                graph.add_edge(test_id, if_id, label="cond", type="control")
            # тело и orelse обрабатываем с копиями локальных определений
            body_defs = dict(local_defs)
            orelse_defs = dict(local_defs)
            # создаём псевдо-узлы для начала тела/orelse (для control-edges)
            body_start = None
            orelse_start = None
            # обрабатываем body
            prev = None
            for stmt in n.body:
                sid = walk(stmt, body_defs)
                if sid is not None:
                    if body_start is None:
                        body_start = sid
                    if prev is not None:
                        graph.add_edge(prev, sid, label="next", type="control")
                    prev = sid
            # обрабатываем orelse
            prev = None
            for stmt in n.orelse:
                sid = walk(stmt, orelse_defs)
                if sid is not None:
                    if orelse_start is None:
                        orelse_start = sid
                    if prev is not None:
                        graph.add_edge(prev, sid, label="next", type="control")
                    prev = sid
            # control-edges: if -> first stmt of body/orelse
            if body_start is not None:
                graph.add_edge(if_id, body_start, label="control_true", type="control")
            if orelse_start is not None:
                graph.add_edge(if_id, orelse_start, label="control_false", type="control")
            # Merge defs: если обе ветви определили одну и ту же переменную — создаём Merge node
            merged = dict(local_defs)
            for var in set(list(body_defs.keys()) + list(orelse_defs.keys())):
                b = body_defs.get(var)
                o = orelse_defs.get(var)
                if b and o and b != o:
                    merge_id = new_id()
                    graph.add_node(merge_id, label=f"Merge:{var}", type="Merge", name=var)
                    graph.add_edge(b, merge_id, label="merge", type="data")
                    graph.add_edge(o, merge_id, label="merge", type="data")
                    merged[var] = merge_id
                elif b:
                    merged[var] = b
                elif o:
                    merged[var] = o
            # обновляем локальные определения вызывающему уровню
            local_defs.clear()
            local_defs.update(merged)
            return if_id

        if isinstance(n, ast.For) or isinstance(n, ast.While):
            loop_id = new_id()
            graph.add_node(loop_id, label=type(n).__name__, type=type(n).__name__)
            if isinstance(n, ast.For):
                target_id = walk(n.target, local_defs)
                iter_id = walk(n.iter, local_defs)
                if iter_id is not None:
                    graph.add_edge(iter_id, loop_id, label="iter", type="control")
                if target_id is not None:
                    graph.add_edge(target_id, loop_id, label="target", type="control")
            else:
                test_id = walk(n.test, local_defs)
                if test_id is not None:
                    graph.add_edge(test_id, loop_id, label="cond", type="control")
            # process body with copy of defs (we don't propagate body defs up to outer scope for simplicity)
            body_defs = dict(local_defs)
            prev = None
            for stmt in n.body:
                sid = walk(stmt, body_defs)
                if sid is not None:
                    if prev is not None:
                        graph.add_edge(prev, sid, label="next", type="control")
                    prev = sid
                    # connect loop header -> first body stmt
                    if prev is not None and loop_id is not None:
                        graph.add_edge(loop_id, prev, label="loop_body", type="control")
            return loop_id

        if isinstance(n, ast.FunctionDef):
            fid = new_id()
            graph.add_node(fid, label=f"Function:{n.name}", type="Function", name=n.name)
            # аргументы — считаются дефами внутри функции
            local = dict(local_defs)
            for arg in n.args.args:
                arg_id = new_id()
                graph.add_node(arg_id, label=f"Arg:{arg.arg}", type="Arg", name=arg.arg)
                local[arg.arg] = arg_id
                graph.add_edge(arg_id, fid, label="arg->func", type="data")
            # тело: connect function node -> first stmt
            first = None
            prev = None
            for stmt in n.body:
                sid = walk(stmt, local)
                if sid is not None:
                    if first is None:
                        first = sid
                    if prev is not None:
                        graph.add_edge(prev, sid, label="next", type="control")
                    prev = sid
            if first is not None:
                graph.add_edge(fid, first, label="func->body", type="control")
            return fid

        if isinstance(n, ast.ClassDef):
            cid = new_id()
            graph.add_node(cid, label=f"Class:{n.name}", type="Class", name=n.name)
            local = dict(local_defs)
            prev = None
            first = None
            for stmt in n.body:
                sid = walk(stmt, local)
                if sid is not None:
                    if first is None:
                        first = sid
                    if prev is not None:
                        graph.add_edge(prev, sid, label="next", type="control")
                    prev = sid
            if first is not None:
                graph.add_edge(cid, first, label="class->body", type="control")
            return cid

        # Generic fallback: пробегаем все поля
        out = None
        for field, value in ast.iter_fields(n):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        out = walk(item, local_defs) or out
            elif isinstance(value, ast.AST):
                out = walk(value, local_defs) or out
        return out

    # начинаем с пустой карты определений
    global_defs = {}
    # Если корень — Module, обрабатываем его тело последовательно и строим control-next между stmt
    if isinstance(tree, ast.Module):
        walk_block(tree.body, global_defs, prev_stmt_node=None)
    else:
        walk(tree, global_defs)

    return graph


def extract_pdg_sequence(code: str) -> str:
    """
    Извлекает упрощённую последовательность узлов PDG для использования как 'текст'.
    Возвращает строку последовательности формата: Type[:name] Type[:name] ...
    """
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()
        graph.graph["next_id"] = 0

        build_pdg_from_ast(tree, graph)

        if len(graph.nodes) == 0:
            return ""

        # Попробуем упорядочить: сначала топологическая сортировка (игнорируем контрольные циклы), иначе по id.
        try:
            topo = list(nx.topological_sort(graph))
        except Exception:
            topo = sorted(graph.nodes())

        seq = []
        for nid in topo:
            node_data = graph.nodes[nid]
            t = node_data.get("type", "Unknown")
            name = node_data.get("name")
            if name:
                seq.append(f"{t}:{name}")
            else:
                seq.append(t)
        # Ограничим длину последовательности (чтобы токенизация была осмысленной)
        return " ".join(seq)
    except SyntaxError as e:
        # синтаксическая ошибка в файле — возвращаем пустую строку
        return ""
    except Exception as e:
        print(f"⚠️ Ошибка построения PDG: {e}")
        return ""


def visualize_pdg_debug(code: str, filename: str):
    """
    Отладочная визуализация PDG — печатает базовую статистику и первые нод-лейблы.
    """
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()
        graph.graph["next_id"] = 0
        build_pdg_from_ast(tree, graph)

        print(f"\n🔍 PDG для {filename}:")
        print(f"Узлов: {len(graph.nodes)}, Рёбер: {len(graph.edges)}")

        # Покажем первые 16 узлов для отладки
        for i, (node_id, node_data) in enumerate(list(graph.nodes(data=True))[:16]):
            label = node_data.get("label", "")
            print(f"  {node_id}: {label}")

        # Покажем примеры рёбер (некоторые)
        print("Примеры ребер (source -> target : label):")
        for i, (u, v, ed) in enumerate(list(graph.edges(data=True))[:20]):
            print(f"  {u} -> {v} : {ed.get('label')} ({ed.get('type')})")

    except Exception as e:
        print(f"Ошибка визуализации PDG: {e}")


# ---------------- 4.1. Функция для получения обратной связи компилятора/исполнения ----------------

def get_compiler_feedback(path: str, run_timeout: int = 5) -> str:
    """
    Пытаемся скомпилировать и выполнить код.
    Возвращаем краткую текстовую последовательность для обучения модели.
    Формат: ключ: сообщение (без длинных стек-трейсов, стараемся укоротить).
    """
    parts = []

    # 1) Компиляция в байткод (py_compile)
    try:
        compile_proc = subprocess.run(
            ["python3", "-m", "py_compile", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=run_timeout
        )
        stderr = compile_proc.stderr.strip()
        if stderr:
            # Обрезаем длинные трассы — оставляем первые 2 строки
            lines = stderr.splitlines()
            brief = " | ".join(lines[:2])
            parts.append(f"COMPILE_ERROR:{brief}")
    except subprocess.TimeoutExpired:
        parts.append("COMPILE_TIMEOUT")
    except Exception as e:
        parts.append(f"COMPILE_CRASH:{str(e)}")

    # 2) Попытка запуска (в изолированном процессе)
    try:
        run_proc = subprocess.run(
            ["python3", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=run_timeout
        )
        stdout = run_proc.stdout.strip()
        stderr = run_proc.stderr.strip()
        if stdout:
            s = stdout.replace("\n", " \\n ")
            # укоротим до 200 символов
            parts.append("RUNTIME_OUT:" + (s[:200] + ("..." if len(s) > 200 else "")))
        if stderr:
            lines = stderr.splitlines()
            brief = " | ".join(lines[:3])
            parts.append("RUNTIME_ERR:" + (brief[:300] + ("..." if len(brief) > 300 else "")))
    except subprocess.TimeoutExpired:
        parts.append("RUNTIME_TIMEOUT")
    except Exception as e:
        parts.append(f"RUNTIME_CRASH:{str(e)}")

    if not parts:
        return "OK"
    return " ".join(parts)


# ---------------- 4. Сбор Python файлов и построение PDG ----------------

# ---------------- 5. Сбор Python файлов и построение PDG + компил. обратной связи ----------------
texts = []
count = 0

print("📁 Сбор Python файлов и построение PDG + сбор компил. фидбека...")
for root, dirs, files in os.walk(CLONE_DIR):
    for file in files:
        if file.endswith(".py") and not file.startswith("test_"):  # Пропускаем тесты
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                pdg_sequence = extract_pdg_sequence(code)

                # фильтр: только осмысленные последовательности (больше N типов)
                if pdg_sequence and len(pdg_sequence.split()) > 20:
                    # собираем компиляторную/рантайм обратную связь
                    compiler_seq = get_compiler_feedback(file_path, run_timeout=3)

                    # Формируем комбинированный обучающий текст
                    # Формат: PDG: ... COMPILER: ...
                    full_text = f"PDG: {pdg_sequence} COMPILER: {compiler_seq}"
                    texts.append({"content": full_text})
                    count += 1

                    # Для первых нескольких файлов — показать отладочную информацию
                    if count <= 3:
                        visualize_pdg_debug(code, file)
                        print(f"Пример последовательности: {pdg_sequence[:200]}...")
                        print(f"Пример compiler_seq: {compiler_seq}")

                if count % 50 == 0 and count > 0:
                    print(f"✅ Обработано {count} файлов")

            except Exception as e:
                print(f"⚠️ Ошибка обработки файла {file_path}: {e}")
                continue

        if count >= LIMIT:
            break
    if count >= LIMIT:
        break

print(f"✅ Собрано {len(texts)} примеров PDG+COMPILER")

if len(texts) == 0:
    print("❌ Нет данных для обучения!")
    print("🔄 Создаем тестовые данные с нормальным PDG...")

    test_codes = [
        """
def calculate_sum(a, b):
    result = a + b
    if result > 10:
        print("Large sum")
        return result * 2
    else:
        print("Small sum")
        return result
        """,

        """
class Calculator:
    def __init__(self, initial_value=0):
        self.value = initial_value

    def add(self, x):
        self.value += x
        return self.value

    def multiply(self, x):
        self.value *= x
        return self.value
        """,

        """
def process_data(data_list):
    results = []
    for item in data_list:
        if item is None:
            continue
        try:
            processed = item * 2
            results.append(processed)
        except Exception as e:
            print(f"Error processing {item}: {e}")
    return results
        """
    ]

    for i, code in enumerate(test_codes):
        seq = extract_pdg_sequence(code)
        if seq:
            texts.append({"content": seq})
            print(f"Тестовый пример {i+1}: {seq[:200]}...")

    print(f"✅ Создано {len(texts)} тестовых примеров с PDG")

# ---------------- 5. Токенизация ----------------
print("🔤 Загрузка токенизатора из локальной папки...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Токенизатор загружен")
except Exception as e:
    print(f"❌ Ошибка загрузки токенизатора: {e}")
    exit(1)

print("🔤 Токенизация данных...")
tokenized_data = []
for i, text in enumerate(texts):
    try:
        tokenized = tokenizer(
            text["content"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized_data.append(tokenized)

        if (i + 1) % 100 == 0:
            print(f"✅ Токенизировано {i + 1} примеров")

    except Exception as e:
        # Пропускаем проблемный пример
        continue

print(f"✅ Токенизировано {len(tokenized_data)} примеров")

if len(tokenized_data) == 0:
    print("❌ Нет токенизированных данных!")
    exit(1)

# ---------------- 7. Создание датасета ----------------
class PDGDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {
            'input_ids': item['input_ids'].squeeze(0),
            'attention_mask': item['attention_mask'].squeeze(0),
            'labels': item['labels'].squeeze(0)
        }

dataset = PDGDataset(tokenized_data)
print("✅ Датасет создан")

# ---------------- 8. Загрузка модели из локальной папки ----------------
print("🤖 Загрузка модели из локальной папки...")
# Попытка использовать Qwen2ForCausalLM как раньше; если у вас другая модель — замените соответствующим классом.
try:
    from transformers import Qwen2ForCausalLM  # если trust_remote_code реализует такой класс
    ModelClass = Qwen2ForCausalLM
except Exception:
    ModelClass = AutoModelForCausalLM  # fallback

try:
    model = ModelClass.from_pretrained(
        LOCAL_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",  # 'cuda' -> используем auto, чтобы корректно распределять
        local_files_only=True
    )
    print("✅ Модель загружена с float16")

except Exception as e:
    print(f"⚠️ Ошибка с float16: {e}")
    try:
        model = ModelClass.from_pretrained(
            LOCAL_MODEL_DIR,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True
        )
        print("✅ Модель загружена без float16")
    except Exception as e2:
        print(f"❌ Ошибка загрузки модели: {e2}")
        exit(1)

# ---------------- 9. Настройка LoRA ----------------
print("🎛️ Настройка LoRA...")
try:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    # Печать сколько параметров обучаемых
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    print("✅ LoRA настроена")

except Exception as e:
    print(f"❌ Ошибка настройки LoRA: {e}")
    exit(1)

# ---------------- 10. Обучение ----------------
print("🚀 Настройка обучения...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=50,
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    fp16=torch.cuda.is_available() and getattr(model, "dtype", None) == torch.float16,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    disable_tqdm=False,
)

def simple_collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=simple_collate_fn
)

print("🚀 Начинаем обучение …")
try:
    trainer.train()
    print("✅ Обучение завершено!")

except Exception as e:
    print(f"❌ Ошибка обучения: {e}")

# ---------------- 11. Сохранение и тестирование ----------------
print("💾 Сохранение модели...")
try:
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Модель сохранена в {OUTPUT_DIR}")

    # Тестируем обученную модель
    print("🧪 Тестируем обученную модель...")
    test_input = "PDG: Function Assign BinOp If Compare Call Return COMPILER: OK"
    test_tokens = tokenizer(test_input, return_tensors="pt", max_length=MAX_LEN, truncation=True)
    if torch.cuda.is_available():
        test_tokens = {k: v.cuda() for k, v in test_tokens.items()}

    with torch.no_grad():
        outputs = model.generate(
            **test_tokens,
            max_length=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 Результат генерации: {generated_text}")

except Exception as e:
    print(f"⚠️ Ошибка сохранения: {e}")

# Очистка памяти
del model, trainer, dataset, tokenized_data
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("🎉 Скрипт завершен!")