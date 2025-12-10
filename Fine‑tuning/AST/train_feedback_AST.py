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
OUTPUT_DIR = "../../ast_feedback_trained_github"
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


def extract_ast_sequence(code: str) -> str:
    """
    Преобразует Python-код в линейное AST-представление
    """
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        return ""
    except Exception as e:
        print(f"Ошибка AST: {e}")
        return ""




def visualize_ast_debug(code: str, filename: str):
    """
    Отладочная печать AST для примера
    """
    try:
        tree = ast.parse(code)
        print(f"\n AST для {filename}:")
        print(ast.dump(tree, indent=2, annotate_fields=True, include_attributes=False)[:500])
    except Exception as e:
        print(f"Ошибка визуализации AST: {e}")


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


# ---------------- 5. Сбор Python файлов + AST + компил. обратная связь ----------------
texts = []
count = 0

print("📁 Сбор Python файлов и построение AST + сбор компил. фидбека...")

for root, dirs, files in os.walk(CLONE_DIR):
    for file in files:
        if file.endswith(".py") and not file.startswith("test_"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                ast_sequence = extract_ast_sequence(code)

                # фильтр: только достаточно большие AST
                if ast_sequence and len(ast_sequence.split()) > 20:

                    compiler_seq = get_compiler_feedback(file_path, run_timeout=3)

                    # Формат для обучения
                    full_text = f"AST: {ast_sequence} COMPILER: {compiler_seq}"
                    texts.append({"content": full_text})
                    count += 1

                    # Отладка для первых файлов
                    if count <= 3:
                        visualize_ast_debug(code, file)
                        print(f"Пример AST sequence: {ast_sequence[:200]}...")
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

print(f"✅ Собрано {len(texts)} примеров AST+COMPILER")

# Если данных нет — создаём тестовые примеры
if len(texts) == 0:
    print("❌ Нет данных для обучения!")
    print("🔄 Создаем тестовые данные с обычным AST...")

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
        seq = extract_ast_sequence(code)
        if seq:
            texts.append({"content": f"AST: {seq}"})
            print(f"Тестовый пример {i+1}: {seq[:200]}...")

    print(f"✅ Создано {len(texts)} тестовых примеров с AST")

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
class ASTDataset(Dataset):
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

dataset = ASTDataset(tokenized_data)
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
    test_input = "AST: Function Assign BinOp If Compare Call Return COMPILER: OK"
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