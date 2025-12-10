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
OUTPUT_DIR = "../../dfg_feedback_trained_github"
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

CLONE_DIR = "../../github_repos"
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

# ---------------- 3. Построение DFG ----------------

def build_dfg_from_ast(node, graph, def_map=None):
    """
    Строит Data Flow Graph (DFG) из AST.
    graph -- networkx.DiGraph(), ноды имеют атрибуты: label, type, names (опционально)
    def_map -- dict: имя переменной -> node_id (последнее определение)
    """
    if def_map is None:
        def_map = {}

    # Инициализация счётчика id в graph.graph чтобы избежать глобов
    if "next_id" not in graph.graph:
        graph.graph["next_id"] = 0

    def new_id():
        nid = graph.graph["next_id"]
        graph.graph["next_id"] += 1
        return nid

    # Рекурсивная обработка, возвращает список node_id, которые соответствуют "выходам" выражения
    def _walk(n, local_def_map):
        # Возвращает список node ids (один или несколько) соответствующих этому узлу (как producer)
        if n is None:
            return []

        # Листовой Node: Name, Constant, etc.
        if isinstance(n, ast.Name):
            nid = new_id()
            label = f"Name:{n.id} ({'Load' if isinstance(n.ctx, ast.Load) else 'Store'})"
            graph.add_node(nid, label=label, type="Name", name=n.id, ctx=type(n.ctx).__name__)
            # Если это чтение — создаём ребро от последнего определения (если есть)
            if isinstance(n.ctx, ast.Load):
                if n.id in local_def_map:
                    graph.add_edge(local_def_map[n.id], nid, label="def->use")
            # Если store — это пока только узел, конкретный assign обработает обновление def_map
            return [nid]

        elif isinstance(n, ast.Constant):
            nid = new_id()
            label = f"Const:{repr(n.value)}"
            graph.add_node(nid, label=label, type="Const")
            return [nid]

        # Binary op
        elif isinstance(n, ast.BinOp):
            left_ids = _walk(n.left, local_def_map)
            right_ids = _walk(n.right, local_def_map)
            op_id = new_id()
            op_name = type(n.op).__name__
            graph.add_node(op_id, label=f"BinOp:{op_name}", type="BinOp")
            # ребра от операндов к операции
            for lid in left_ids + right_ids:
                graph.add_edge(lid, op_id, label="operand")
            return [op_id]

        elif isinstance(n, ast.UnaryOp):
            operand_ids = _walk(n.operand, local_def_map)
            op_id = new_id()
            op_name = type(n.op).__name__
            graph.add_node(op_id, label=f"UnaryOp:{op_name}", type="UnaryOp")
            for oid in operand_ids:
                graph.add_edge(oid, op_id, label="operand")
            return [op_id]

        elif isinstance(n, ast.Call):
            func_ids = _walk(n.func, local_def_map)
            arg_ids = []
            for a in n.args:
                arg_ids.extend(_walk(a, local_def_map))
            call_id = new_id()
            graph.add_node(call_id, label="Call", type="Call")
            for fid in func_ids + arg_ids:
                graph.add_edge(fid, call_id, label="arg")
            # keywords
            for kw in getattr(n, "keywords", []):
                arg_ids = _walk(kw.value, local_def_map)
                for aid in arg_ids:
                    graph.add_edge(aid, call_id, label=f"kw:{kw.arg}")
            return [call_id]

        elif isinstance(n, ast.Assign):
            # RHS — value
            value_ids = _walk(n.value, local_def_map)
            assign_id = new_id()
            graph.add_node(assign_id, label="Assign", type="Assign")
            # ребра от value к assign
            for vid in value_ids:
                graph.add_edge(vid, assign_id, label="value")
            # Для каждого target — если это Name, создаём узел def и edge assign->def и обновляем def_map
            new_defs = []
            for t in n.targets:
                if isinstance(t, ast.Name):
                    def_id = new_id()
                    graph.add_node(def_id, label=f"Def:{t.id}", type="Def", name=t.id)
                    graph.add_edge(assign_id, def_id, label="assign->def")
                    # ребро от последнего определения (если есть) не требуется — dataflow от value
                    # Обновляем локальную карту определений
                    local_def_map = dict(local_def_map)  # клонируем чтобы изменения не ударяли вверх по стеку
                    local_def_map[t.id] = def_id
                    new_defs.append(def_id)
                else:
                    # Если target сложнее (tuple, attr, subscript) — обрабатываем рекурсивно
                    target_outs = _walk(t, local_def_map)
                    for tout in target_outs:
                        graph.add_edge(assign_id, tout, label="assign->target")
                        # если у target есть имя внутри, можно попытаться обновить def_map — но для простоты пропустим
            return new_defs if new_defs else [assign_id]

        elif isinstance(n, ast.AugAssign):
            # target op= value  -> читаем old def, создаём op node, и новый def
            target_ids = _walk(n.target, local_def_map)
            value_ids = _walk(n.value, local_def_map)
            op_id = new_id()
            op_name = type(n.op).__name__
            graph.add_node(op_id, label=f"AugOp:{op_name}", type="AugOp")
            for tid in target_ids + value_ids:
                graph.add_edge(tid, op_id, label="operand")
            # создаём новый деф для target if it's Name
            if isinstance(n.target, ast.Name):
                def_id = new_id()
                graph.add_node(def_id, label=f"Def:{n.target.id}", type="Def", name=n.target.id)
                graph.add_edge(op_id, def_id, label="result")
                local_def_map = dict(local_def_map)
                local_def_map[n.target.id] = def_id
                return [def_id]
            return [op_id]

        elif isinstance(n, ast.If):
            # Для DFG: обрабатываем test, body и orelse. Тест — producer для веток
            test_ids = _walk(n.test, local_def_map)
            if_id = new_id()
            graph.add_node(if_id, label="If", type="If")
            for tid in test_ids:
                graph.add_edge(tid, if_id, label="cond")
            # Обработка тела — используем копию def_map (ветви могут определять разные переменные)
            body_def_map = dict(local_def_map)
            for stmt in n.body:
                _walk(stmt, body_def_map)
            orelse_def_map = dict(local_def_map)
            for stmt in n.orelse:
                _walk(stmt, orelse_def_map)
            merged_map = dict(local_def_map)
            for var in set(list(body_def_map.keys()) + list(orelse_def_map.keys())):
                b = body_def_map.get(var)
                o = orelse_def_map.get(var)
                if b and o and b != o:
                    merge_id = new_id()
                    graph.add_node(merge_id, label=f"Merge:{var}", type="Merge", name=var)
                    graph.add_edge(b, merge_id, label="merge")
                    graph.add_edge(o, merge_id, label="merge")
                    merged_map[var] = merge_id
                elif b:
                    merged_map[var] = b
                elif o:
                    merged_map[var] = o
            return [if_id], merged_map

        elif isinstance(n, ast.For) or isinstance(n, ast.While):
            # Аналогично: производим nodes для условия/итератора и тела
            loop_id = new_id()
            graph.add_node(loop_id, label=type(n).__name__, type=type(n).__name__)
            if isinstance(n, ast.For):
                target_ids = _walk(n.target, local_def_map)
                iter_ids = _walk(n.iter, local_def_map)
                for iid in iter_ids + target_ids:
                    graph.add_edge(iid, loop_id, label="iter")
            else:
                test_ids = _walk(n.test, local_def_map)
                for tid in test_ids:
                    graph.add_edge(tid, loop_id, label="cond")
            body_def_map = dict(local_def_map)
            for stmt in n.body:
                _walk(stmt, body_def_map)
            return [loop_id]

        elif isinstance(n, ast.Return):
            val_ids = _walk(n.value, local_def_map) if n.value else []
            ret_id = new_id()
            graph.add_node(ret_id, label="Return", type="Return")
            for vid in val_ids:
                graph.add_edge(vid, ret_id, label="ret")
            return [ret_id]

        elif isinstance(n, ast.FunctionDef):
            fid = new_id()
            graph.add_node(fid, label=f"Function:{n.name}", type="Function", name=n.name)
            # аргументы — считаются определениями внутри функции
            local_map = dict(local_def_map)
            for arg in n.args.args:
                arg_id = new_id()
                graph.add_node(arg_id, label=f"Arg:{arg.arg}", type="Arg", name=arg.arg)
                local_map[arg.arg] = arg_id
                graph.add_edge(arg_id, fid, label="arg->func")
            # тело
            for stmt in n.body:
                _walk(stmt, local_map)
            return [fid]

        elif isinstance(n, ast.ClassDef):
            cid = new_id()
            graph.add_node(cid, label=f"Class:{n.name}", type="Class", name=n.name)
            local_map = dict(local_def_map)
            for stmt in n.body:
                _walk(stmt, local_map)
            return [cid]

        elif isinstance(n, ast.Expr):
            return _walk(n.value, local_def_map)

        elif isinstance(n, ast.Compare):
            left_ids = _walk(n.left, local_def_map)
            comp_id = new_id()
            graph.add_node(comp_id, label="Compare", type="Compare")
            for lid in left_ids:
                graph.add_edge(lid, comp_id, label="left")
            for op, comp in zip(n.ops, n.comparators):
                comp_ids = _walk(comp, local_def_map)
                for cid in comp_ids:
                    graph.add_edge(cid, comp_id, label="comp")
            return [comp_id]

        else:
            out_ids = []
            for field, value in ast.iter_fields(n):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            out_ids.extend(_walk(item, local_def_map))
                elif isinstance(value, ast.AST):
                    out_ids.extend(_walk(value, local_def_map))
            return out_ids

    # Запускаем обход с пустой картой определений
    _walk(node, def_map)
    return graph


def extract_dfg_sequence(code: str) -> str:
    """
    Извлекает последовательность узлов DFG из кода.
    Возвращает строку последовательности типов нод (например: Def Assign Call Name ...)
    """
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()
        graph.graph["next_id"] = 0

        build_dfg_from_ast(tree, graph, def_map={})

        if len(graph.nodes) == 0:
            return ""

        try:
            topo = list(nx.topological_sort(graph))
        except Exception:
            # fallback: по порядку добавления (id)
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
        return " ".join(seq)

    except SyntaxError as e:
        print(f"⚠️ Синтаксическая ошибка в коде: {e}")
        return ""
    except Exception as e:
        print(f"⚠️ Ошибка построения DFG: {e}")
        return ""



def visualize_dfg_debug(code: str, filename: str):
    """
    Отладочная визуализация DFG — печатает базовую статистику и первые нод-лейблы.
    (Не рисует графику, просто вывод в консоль для больших репозиториев)
    """
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()
        graph.graph["next_id"] = 0
        build_dfg_from_ast(tree, graph, def_map={})

        print(f"\n🔍 DFG для {filename}:")
        print(f"Узлов: {len(graph.nodes)}, Рёбер: {len(graph.edges)}")

        # Покажем первые 12 узлов для отладки
        for i, (node_id, node_data) in enumerate(list(graph.nodes(data=True))[:12]):
            label = node_data.get("label", "")
            print(f"  {node_id}: {label}")

    except Exception as e:
        print(f"Ошибка визуализации DFG: {e}")



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



# ---------------- 5. Сбор Python файлов и построение DFG + компил. обратной связи ----------------
texts = []
count = 0

print("📁 Сбор Python файлов и построение DFG + сбор компил. фидбека...")
for root, dirs, files in os.walk(CLONE_DIR):
    for file in files:
        if file.endswith(".py") and not file.startswith("test_"):  # Пропускаем тесты
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                dfg_sequence = extract_dfg_sequence(code)

                # фильтр: только осмысленные последовательности (больше N типов)
                if dfg_sequence and len(dfg_sequence.split()) > 20:
                    # собираем компиляторную/рантайм обратную связь
                    compiler_seq = get_compiler_feedback(file_path, run_timeout=3)

                    # Формируем комбинированный обучающий текст
                    # Формат: DFG: ... COMPILER: ...
                    full_text = f"DFG: {dfg_sequence} COMPILER: {compiler_seq}"
                    texts.append({"content": full_text})
                    count += 1

                    # Для первых нескольких файлов — показать отладочную информацию
                    if count <= 3:
                        visualize_dfg_debug(code, file)
                        print(f"Пример последовательности: {dfg_sequence[:200]}...")
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

print(f"✅ Собрано {len(texts)} примеров DFG+COMPILER")

if len(texts) == 0:
    print("❌ Нет данных для обучения!")
    print("🔄 Создаем тестовые данные с нормальным DFG...")

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
        seq = extract_dfg_sequence(code)
        if seq:
            texts.append({"content": seq})
            print(f"Тестовый пример {i+1}: {seq[:200]}...")

    print(f"✅ Создано {len(texts)} тестовых примеров с DFG")

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
class DFGDataset(Dataset):
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

dataset = DFGDataset(tokenized_data)
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
    test_input = "DFG: Function Assign BinOp If Compare Call Return COMPILER: OK"
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