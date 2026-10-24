from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from models.hf_data import HFParquetSource, resolve_revision

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUALITY_REMOVALS = Path(__file__).resolve().parent / "removal_lists" / "train_quality_removals.parquet"
BATCH_SIZE = 100000


def remove_audited_rows(train: pl.LazyFrame) -> pl.LazyFrame:
    if not QUALITY_REMOVALS.is_file():
        raise FileNotFoundError(f"Не найден список очистки: {QUALITY_REMOVALS}")
    removals = (
        pl.scan_parquet(QUALITY_REMOVALS)
        .select("question_clean", "code")
        .unique()
    )
    return train.join(
        removals,
        on=["question_clean", "code"],
        how="anti",
        maintain_order="left",
    )

# Unicode, переносы и пробелы по краям
def clean_normalization(q):
    return q.str.normalize("NFC").str.replace_all(r"\r\n?", "\n").str.strip_chars()

# Внешняя ограда Markdown
def clean_markdown(q):
    return (
        q.str.replace(
            r"(?i)^\s*```(?:(?:text|markdown|md|python|py|math|plaintext)[ \t]*(?:\r?\n|$)|[ \t]*(?:\r?\n)?)", "",
        )
        .str.replace(r"[ \t]*```\s*$", "")
        .str.strip_chars()
    )

# Внешнее жирное выделение Markdown
def clean_bold(q):
    return q.str.replace(r"(?s)^\*\*(\p{L}.*?)\*\*$", "$1")

# Нумерация перед условием
def clean_list_numbers(q):
    return q.str.replace(r"^\s*(?:(?:\d{1,3}[.)]|\(\d{1,3}\)|\[\d{1,3}\])\s+)+"
        r"(\p{L}|[-*#_`]+\s*\p{L}|<(?:sep>|hr>|/pre>|b>|start of question|\|)|[\ufeff\u200b])", "$1")

# Доп функция для нумерации
def strip_label(q, name):
    return q.str.replace(
        r"(?i)^\s*[-*#_`]*\s*(?:" + name + r")"
        r"(?:[ \t]*#?[ \t]*(?:\d+|x(?:[ \t]*[+-][ \t]*\d+)?|"
        r"one|two|three|four|five|six|seven|eight|nine|ten|[a-z]))?"
        r"[ \t]*[*_`]*[ \t]*[:]\s*(?:(?:\*{2}|_{2}|`)(?:\s+|$))?", "",
    )

# Student
def clean_student(q):
    return strip_label(q, r"student")

# Answer
def clean_answer(q):
    return strip_label(q, r"(?:final\s+)?answers?")

# Solution
def clean_solution(q):
    return strip_label(q, r"(?:final\s+)?solutions?(?:\s+(?:example|to\s+(?:one|each)\s+question))?")

# Result
def clean_result(q):
    return strip_label(q, r"(?:final\s+)?results?(?:\s+(?:of\s+(?:the\s+)?(?:python\s+)?program|should\s+be|is|in\s+this\s+case|example))?")

# Question
def clean_question_prefix(q):
    return (
        strip_label(q, r"questions?")
        .str.replace(r"(?i)^\s*q\s*#?\d+\s*(?:[:]\s*|[.)]\s+)", "")
        .str.replace(r"(?i)^\s*[-*#_`]*\s*questions?\s*#?\s*\d+\s*[.)]\s+(\p{L})", "$1")
        .str.replace(r"(?i)^\s*\*\*questions?\s*#?\s*\d+\*\*\s+", "")
    )

# Problem
def clean_problem(q):
    return strip_label(q, r"problems?")

# Template
def clean_template(q):
    return strip_label(q, r"templates?")

# Scenario
def clean_scenario(q):
    return strip_label(q, r"scenarios?")

# Code / Python code
def clean_code(q):
    return strip_label(q, r"(?:python\s+)?codes?")

# Program / Python program
def clean_program(q):
    return strip_label(q, r"(?:python\s+)?programs?")

# Possible
def clean_possible(q):
    return q.str.replace(
        r"(?i)^[\s\-*=._#]*"
        r"(?:(?:(?:here|below)\s+(?:are|is)\s+(?:(?:\d+|some)\s+)?)|s?ome\s+(?:other\s+)?)?"
        + possible_header, "",
    )

# Possible template
def clean_possible_template(q):
    return q.str.replace(r"(?i)^\s*possible\s+template\s*[:]\s*", "")

# Инструкция о Question 0
def clean_example_instruction(q):
    return q.str.replace(r"(?i)^\s*question\s*0\s+is\s+already\s+given\s+as\s+an\s+example\.\s*", "")

# Инструкция о языке
def clean_english_instruction(q):
    return q.str.replace(
        r"(?i)^[\s*-]*questions and programs should be written in\s+\*{0,2}english\*{0,2}\s*[.:;]?\s*", "",
    )

# Инструкция не копировать исходный пример
def clean_copy_instruction(q):
    return q.str.replace(
        r"(?i)^[\s*-]*avoid\s+using\s+the\s+same\s+"
        r"(?:numbers|words|names|units)(?:\s+or\s+(?:numbers|words|names|units))*"
        r"\s+as\s+(?:the\s+)?original\s+question\.\s*", "",
    )

# Вводная инструкция перед Possible
possible_header = (
    r"possible\s+(?:\d+\s+)?"
    r"(?:(?:new|additional|math|similar|sample|other|example|alternative|comparable|rewritten|addition|follow-up)\s+)*"
    r"(?:(?:examples?|set|formats?)\s+(?:of|for)\s+(?:the\s+)?(?:new\s+)?)?"
    r"(?:questions?|answers?|examples?|formats?|outputs?|rewrite|problems?|solutions?)(?:\s+\d+)?"
    r"(?:\s*(?:,\s*(?:and\s+)?|\s+(?:and|with)\s+)"
    r"(?:(?:their|corresponding|python|new)\s+)*(?:questions?|programs?|codes?|answers?|solutions?|outputs?|results?))*"
    r"(?:\s+similar\s+to\s+(?:the\s+)?(?:original|given|above|example)(?:\s+(?:one|question))?)?"
    r"(?:\s+are)?\s*[:]\s*"
)

def clean_possible_instruction(q):
    return q.str.replace(
        r"(?i)^[\s\-*#=._]*"
        r"(?:next\s+question\b|etc\.|see\s+example\b|avoid\s+using\b"
        r"|start\s+with\b|put\s+each\b|repeat\b|new\s+line\b|before\s+starting\b"
        r"|you\s+(?:may|can)\s+use\b|each\s+question\s+should\b"
        r"|refer\s+to\s+this\s+example\b|feel\s+free\s+to\s+use\b)"
        r"[^?\n]*?((?:(?:here|below)\s+(?:are|is)\s+(?:(?:\d+|some)\s+)?|s?ome\s+(?:other\s+)?)?"
        + possible_header + r")", "$1",
    )

# Служебные токены разметки
def clean_special_tokens(q):
    return q.str.replace(
        r"(?i)^\s*(?:<\|(?:startoftext|endofanswer|seperate|separate|end|end of cell)\|>"
        r"|<\|\|>|<sep>|<hr>|</pre>|<b>\s*</b>|<start of question\s+\d+>)\s*", "",
    )

# Экранированный перенос
def clean_escaped_prefix(q):
    return (
        q.str.replace(r"^\s*\\n(?:\s+|(\p{Lu})|$)", "$1")
        .str.replace(r"^\s*\\end\{verbatim\}\s*", "")
    )

# Метка @question
def clean_at_question(q):
    return q.str.replace(r"(?i)^\s*@question\s*\d+\s*[:]\s*", "")

# Метка [NLP …]
def clean_nlp_label(q):
    return q.str.replace(r"(?i)^\s*\[nlp\s*-\s*[\p{L} -]+\]\s*", "")

# Служебная метка в круглых скобках
def clean_parenthetical_label(q):
    return q.str.replace(
        r"(?i)^\s*\((?:question|input is in bold|without the solution|"
        r"end of problem\s+\d+|beginning of problem\s+\d+)\)\s*", "",
    )

# Заголовок Markdown
def clean_heading(q):
    return q.str.replace(r"^\s*#{2,6}[ \t]+(\p{Lu})", "$1")

# Маркер списка перед текстом
def clean_bullet(q):
    return q.str.replace(r"^\s*(?:[-—]\s+|\*+\s+)(\p{Lu}\p{Ll}+\b|A\b)", "$1")

# Разделитель Markdown перед текстом
def clean_separator(q):
    return q.str.replace(r"^\s*(?:-{3,}|={3,}|_{3,}|\*{3,})\s+(\p{L}|\d|-\s+\p{Lu})", "$1")

# Многоточие перед текстом
def clean_dots(q):
    return q.str.replace(r"^\s*(?:\.{2,}|(?:\.\s+)+)\s*(\p{Lu}\p{Ll}+\b|A\b)", "$1")

# BOM / zero-width в начале
def clean_invisible_prefix(q):
    return q.str.strip_chars_start("\ufeff\u200b")

# Хвост с незаполненными присваиваниями
def clean_placeholder_suffix(q):
    return q.str.replace(
        r"(?i)(\?)\s*(?:[a-z_][a-z_0-9]*\s*=\s*(?:\?(?:\s*\?)*|\.\.\.)\s*)+$", "$1",
    )

# Числовой хвост Answer
def clean_answer_suffix(q):
    return q.str.replace(
        r"(?i)(\?)\s*[*_`#]*\s*(?:final\s+)?answers?\s*[*_`]*\s*:\s*"
        r"(?:[$€£]\s*)?[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*[.!]?\s*$", "$1",
    )

# Числовой хвост Result
def clean_result_suffix(q):
    return q.str.replace(
        r"(?i)(\?)\s*[*_`#]*\s*(?:final\s+)?results?\s*[*_`]*\s*:\s*"
        r"(?:[$€£]\s*)?[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*[.!]?\s*$", "$1",
    )

# Инструкция о формате Python
def clean_python_instruction(q):
    return q.str.replace(
        r"(?i)^[\s\-*#_`]*(?:all[*_`]*\s+)?the\s+variable\s+names\s+and\s+the\s+code\s+"
        r"(?:need|have)\s+to\s+be\s+in\s+the\s+format\s+of\s+python[!.\s]+", "",
    )

# Незаполненный Result XXX
def clean_result_placeholder(q):
    return q.str.replace(
        r"(?i)^\s*xxx\s*\(\s*replace\s+xxx\s+with\s+(?:the\s+)?actual\s+result\s+of\s+"
        r"(?:the\s+)?python\s+program\s*\)\s*", "",
    )

# Маркер // перед текстом
def clean_slash_prefix(q):
    return q.str.replace(r"^\s*//\s+(\p{Lu}\p{Ll}+\b|A\b|\d{1,3}[.)]\s+\p{Lu})", "$1")

# Двоеточие перед текстом
def clean_colon_prefix(q):
    return q.str.replace(r"^\s*:\s*(\p{Lu}\p{Ll}+\b|A\b)", "$1")

# Цитатный маркер Markdown
def clean_blockquote(q):
    return q.str.replace(r"^\s*>+\s*(\p{Lu}\p{Ll}+\b|A\b)", "$1")

# Полный проход
def apply_cleaning_pass(q):
    q = clean_normalization(q)
    q = clean_markdown(q)
    q = clean_bold(q)
    q = clean_list_numbers(q)
    q = clean_student(q)
    q = clean_answer(q)
    q = clean_solution(q)
    q = clean_result(q)
    q = clean_question_prefix(q)
    q = clean_problem(q)
    q = clean_template(q)
    q = clean_scenario(q)
    q = clean_code(q)
    q = clean_program(q)
    q = clean_possible(q)
    q = clean_possible_template(q)
    q = clean_example_instruction(q)
    q = clean_english_instruction(q)
    q = clean_copy_instruction(q)
    q = clean_possible_instruction(q)
    q = clean_special_tokens(q)
    q = clean_escaped_prefix(q)
    q = clean_at_question(q)
    q = clean_nlp_label(q)
    q = clean_parenthetical_label(q)
    q = clean_heading(q)
    q = clean_bullet(q)
    q = clean_separator(q)
    q = clean_dots(q)
    q = clean_invisible_prefix(q)
    q = clean_placeholder_suffix(q)
    q = clean_answer_suffix(q)
    q = clean_result_suffix(q)
    q = clean_python_instruction(q)
    q = clean_result_placeholder(q)
    q = clean_slash_prefix(q)
    q = clean_colon_prefix(q)
    q = clean_blockquote(q)
    return q


def clean_questions(source, path, progress):
    schema = source.collect_schema()
    schema["question_clean"] = pl.String
    with pq.ParquetWriter(
        path, pl.DataFrame(schema=schema).to_arrow().schema,
        compression="zstd", use_dictionary=False,
    ) as writer:
        for batch in source.collect_batches(chunk_size=BATCH_SIZE, maintain_order=True):
            batch = batch.with_columns(pl.col("question").alias("question_clean"))
            while True:
                before = batch["question_clean"]
                batch = batch.with_columns(
                    apply_cleaning_pass(pl.col("question_clean")).alias("question_clean")
                )
                if batch["question_clean"].equals(before):
                    break
            writer.write_table(batch.to_arrow())
            progress.update(batch.height)


def main():
    repo = "SlavaYus/tinysms_qa_12m"
    revision = resolve_revision(repo)
    train = HFParquetSource(
        repo, [f"data/train/part-{index:05d}.parquet" for index in range(60)],
        revision, ["question", "code"],
    )
    test = HFParquetSource(
        repo, ["data/test/test.parquet"], revision, ["question"],
    )
    total = train.total_rows + test.total_rows

    with TemporaryDirectory(prefix=".question-preprocessing-", dir=DATA_DIR) as temp:
        temp = Path(temp)
        with tqdm(total=total, desc="Очистка train", unit="строк", dynamic_ncols=True) as progress:
            clean_questions(train, temp / "train.parquet", progress)
            progress.set_description_str("Очистка test")
            clean_questions(test, temp / "test.parquet", progress)

        train = pl.scan_parquet(temp / "train.parquet")
        test = pl.scan_parquet(temp / "test.parquet")

        # Пустое условие train
        train = train.filter(
            pl.col("question_clean").is_not_null()
            & pl.col("question_clean").str.strip_chars().ne("")
        )

        train = train.filter(~pl.col("question_clean").str.contains(r"^[\s.*_`#=\-]+$"))

        # Пустая программа train
        train = train.filter(
            pl.col("code").is_not_null() & pl.col("code").str.strip_chars().ne("")
        )

        # Пересечение train с test по очищенному вопросу
        train = train.join(
            test.select("question_clean").unique(),
            on="question_clean", how="anti", maintain_order="left",
        )

        # Проверенные аудитом строки: 161 003 AST-дубликата,
        # 7 неопределённых имён и 931 несогласованное решение.
        # 214 строк входят сразу в две группы, поэтому всего удаляется 161 727 строк.
        train = remove_audited_rows(train)

        # Полные дубликаты train
        train = train.unique(subset=["question_clean", "code"], keep="first", maintain_order=True)

        with tqdm(total=2, desc="Фильтрация и сохранение", unit="файл", dynamic_ncols=True) as progress:
            train.sink_parquet(temp / "train_clean.parquet", engine="streaming")
            progress.update(1)
            test.sink_parquet(temp / "test_clean.parquet", engine="streaming")
            progress.update(1)

        (temp / "train_clean.parquet").replace(DATA_DIR / "train_clean.parquet")
        (temp / "test_clean.parquet").replace(DATA_DIR / "test_clean.parquet")


if __name__ == "__main__":
    main()
