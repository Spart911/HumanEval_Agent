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
OUTPUT_DIR = "../../cfg_feedback_trained_github"
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

# ---------------- 3. Построение CFG ----------------

def build_cfg_from_ast(node, graph, parent=None, edge_label=""):
    """
    Строит настоящий Control Flow Graph из AST дерева
    """
    if not isinstance(node, ast.AST):
        return

    current_id = len(graph.nodes)
    node_name = type(node).__name__

    # Добавляем информацию о позиции для отладки
    if hasattr(node, 'lineno'):
        node_label = f"{node_name} (line {node.lineno})"
    else:
        node_label = node_name

    graph.add_node(current_id, label=node_label, type=node_name)

    # Добавляем ребро от родителя
    if parent is not None:
        graph.add_edge(parent, current_id, label=edge_label)

    # Обрабатываем разные типы узлов
    if isinstance(node, ast.FunctionDef):
        # Для функций добавляем аргументы и тело
        for arg in node.args.args:
            build_cfg_from_ast(arg, graph, current_id, "arg")
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "body")

    elif isinstance(node, ast.ClassDef):
        # Для классов добавляем тело
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "class_body")

    elif isinstance(node, ast.If):
        # Для условий добавляем test, body и orelse
        build_cfg_from_ast(node.test, graph, current_id, "condition")
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "if_body")
        for item in node.orelse:
            build_cfg_from_ast(item, graph, current_id, "else_body")

    elif isinstance(node, ast.For) or isinstance(node, ast.While):
        # Для циклов добавляем target, iter и body
        build_cfg_from_ast(node.target, graph, current_id, "loop_var")
        build_cfg_from_ast(node.iter, graph, current_id, "loop_iter")
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "loop_body")
        for item in node.orelse:
            build_cfg_from_ast(item, graph, current_id, "loop_else")

    elif isinstance(node, ast.Try):
        # Для try-except блоков
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "try_body")
        for handler in node.handlers:
            build_cfg_from_ast(handler, graph, current_id, "except")
        for item in node.orelse:
            build_cfg_from_ast(item, graph, current_id, "else_body")
        for item in node.finalbody:
            build_cfg_from_ast(item, graph, current_id, "finally")

    elif isinstance(node, ast.ExceptHandler):
        # Для обработчиков исключений
        if node.type:
            build_cfg_from_ast(node.type, graph, current_id, "exception_type")
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "except_body")

    elif isinstance(node, ast.With):
        # Для with блоков
        for item in node.items:
            build_cfg_from_ast(item, graph, current_id, "with_item")
        for item in node.body:
            build_cfg_from_ast(item, graph, current_id, "with_body")

    elif isinstance(node, ast.Call):
        # Для вызовов функций
        build_cfg_from_ast(node.func, graph, current_id, "function")
        for arg in node.args:
            build_cfg_from_ast(arg, graph, current_id, "arg")
        for keyword in node.keywords:
            build_cfg_from_ast(keyword, graph, current_id, "kwarg")

    elif isinstance(node, ast.BinOp):
        # Для бинарных операций
        build_cfg_from_ast(node.left, graph, current_id, "left")
        build_cfg_from_ast(node.op, graph, current_id, "operator")
        build_cfg_from_ast(node.right, graph, current_id, "right")

    elif isinstance(node, ast.Compare):
        # Для операций сравнения
        build_cfg_from_ast(node.left, graph, current_id, "left")
        for op, comparator in zip(node.ops, node.comparators):
            build_cfg_from_ast(op, graph, current_id, "comparator_op")
            build_cfg_from_ast(comparator, graph, current_id, "comparator")

    elif isinstance(node, ast.Assign):
        # Для присваиваний
        for target in node.targets:
            build_cfg_from_ast(target, graph, current_id, "target")
        build_cfg_from_ast(node.value, graph, current_id, "value")

    elif isinstance(node, ast.AugAssign):
        # Для augmented присваиваний
        build_cfg_from_ast(node.target, graph, current_id, "target")
        build_cfg_from_ast(node.op, graph, current_id, "operator")
        build_cfg_from_ast(node.value, graph, current_id, "value")

    else:
        # Для всех остальных узлов обрабатываем дочерние элементы
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        build_cfg_from_ast(item, graph, current_id, field)
            elif isinstance(value, ast.AST):
                build_cfg_from_ast(value, graph, current_id, field)


def extract_cfg_sequence(code: str) -> str:
    """
    Извлекает последовательность узлов CFG из кода
    """
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()

        # Строим CFG
        build_cfg_from_ast(tree, graph)

        if len(graph.nodes) == 0:
            return ""

        # Извлекаем последовательность узлов в порядке обхода
        sequence = []

        def traverse_cfg(node_id, visited=None):
            if visited is None:
                visited = set()

            if node_id in visited:
                return

            visited.add(node_id)
            node_data = graph.nodes[node_id]
            sequence.append(node_data['type'])  # Используем только тип узла

            # Обходим соседей в порядке out-edges
            for neighbor in graph.successors(node_id):
                traverse_cfg(neighbor, visited)

        # Начинаем обход с корневого узла (0)
        traverse_cfg(0)

        return " ".join(sequence)

    except SyntaxError as e:
        print(f"Синтаксическая ошибка в коде: {e}")
        return ""
    except Exception as e:
        print(f"Ошибка построения CFG: {e}")
        return ""


def visualize_cfg_debug(code: str, filename: str):
    """
    Функция для отладки - визуализирует CFG для небольшого кода
    """
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()
        build_cfg_from_ast(tree, graph)

        print(f"\n🔍 CFG для {filename}:")
        print(f"Узлов: {len(graph.nodes)}, Ребер: {len(graph.edges)}")

        # Покажем первые 10 узлов для отладки
        for i, (node_id, node_data) in enumerate(list(graph.nodes(data=True))[:10]):
            print(f"  {node_id}: {node_data['label']}")

    except Exception as e:
        print(f"Ошибка визуализации: {e}")



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



# ---------------- 5. Сбор Python файлов и построение CFG + компил. обратной связи ----------------
texts = []
count = 0

print("📁 Сбор Python файлов и построение CFG + сбор компил. фидбека...")
for root, dirs, files in os.walk(CLONE_DIR):
    for file in files:
        if file.endswith(".py") and not file.startswith("test_"):  # Пропускаем тесты
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                cfg_sequence = extract_cfg_sequence(code)

                # фильтр: только осмысленные последовательности (больше N типов)
                if cfg_sequence and len(cfg_sequence.split()) > 20:
                    # собираем компиляторную/рантайм обратную связь
                    compiler_seq = get_compiler_feedback(file_path, run_timeout=3)

                    # Формируем комбинированный обучающий текст
                    # Формат: CFG: ... COMPILER: ...
                    full_text = f"CFG: {cfg_sequence} COMPILER: {compiler_seq}"
                    texts.append({"content": full_text})
                    count += 1

                    # Для первых нескольких файлов — показать отладочную информацию
                    if count <= 3:
                        visualize_cfg_debug(code, file)
                        print(f"Пример последовательности: {cfg_sequence[:200]}...")
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

print(f"✅ Собрано {len(texts)} примеров CFG+COMPILER")

if len(texts) == 0:
    print("❌ Нет данных для обучения!")
    print("🔄 Создаем тестовые данные с нормальным CFG...")

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
        seq = extract_cfg_sequence(code)
        if seq:
            texts.append({"content": seq})
            print(f"Тестовый пример {i+1}: {seq[:200]}...")

    print(f"✅ Создано {len(texts)} тестовых примеров с CFG")

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
class CFGDataset(Dataset):
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

dataset = CFGDataset(tokenized_data)
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
    test_input = "CFG: Function Assign BinOp If Compare Call Return COMPILER: OK"
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