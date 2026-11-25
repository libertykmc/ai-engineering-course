"""
Модуль 4, Занятие 8 — оценка качества и защита RAG-сценария.

Реализованы classify, llm_judge, sweep_chunking и resolve_conflicts.
Контрольный набор оценивается по разметке и независимым вызовом GigaChat.

Занятие 7 научило rag_assistant.py отвечать с источником и отказываться
при низкой уверенности. Здесь мы сравниваем два способа проверки:

  1. По разметке eval-набора — быстро, дёшево, воспроизводимо.
  2. Через LLM-as-Judge — гибко и наглядно: GigaChat видит вопрос,
     контекст поиска и ответ ассистента и объясняет, что именно не так.

Запуск (из папки module_04, индекс должен быть собран):
    python evaluate_rag.py                  # сравнение двух способов оценки
    python evaluate_rag.py --sweep          # перебор CHUNK_SIZE/OVERLAP
    python evaluate_rag.py --conflict       # демо конфликта версий регламента
    python evaluate_rag.py --embedded       # без Docker
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime

import httpx
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.exceptions import RateLimitError, ServerError
from sentence_transformers import SentenceTransformer

import ingestion
from qdrant_connect import make_client
from vector_store import MODEL_NAME, load_manifest, search
from rag_assistant import answer, format_context, REFUSAL_TEXT

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "qdrant_chunks"
CORPUS_DIR = BASE_DIR / "data" / "corpus"
CONFLICT_DIR = BASE_DIR / "data" / "corpus_conflict"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

QUESTIONS = json.loads(
    (BASE_DIR / "data" / "questions.json").read_text(encoding="utf-8")
)["questions"]

# Комбинации для перебора — те же, что в Занятии 6, чтобы результаты
# было с чем сравнивать. Пересборка индекса на каждую комбинацию занимает
# несколько минут: --sweep запускай отдельно, не в общем прогоне.
CHUNK_COMBOS = [(1000, 200), (600, 0), (600, 150), (400, 0), (400, 100), (300, 150)]

_SEP  = "─" * 88
_SEP2 = "═" * 88

# Системный промпт судьи. Менять не нужно — он уже настроен на корпус курса.
JUDGE_SYSTEM = (
    "Ты — независимый оценщик качества RAG-ассистента. "
    "Ассистент отвечает на вопросы по корпусу производственных документов "
    "промышленного предприятия.\n\n"
    "Тебе даются три части:\n"
    "  ВОПРОС — то, что спросил пользователь.\n"
    "  КОНТЕКСТ — фрагменты документов, которые нашёл поиск и передал ассистенту.\n"
    "  ОТВЕТ — что ответил ассистент.\n\n"
    "Оцени ответ по трём критериям:\n"
    "  1. answer_relevance — отвечает ли ответ именно на вопрос пользователя.\n"
    "  2. context_relevance — действительно ли найденный контекст подходит для этого вопроса.\n"
    "  3. answer_faithfulness — не выдумывает ли ответ факты сверх переданного контекста.\n\n"
    "Для каждого критерия используй одно из значений: high, medium, low.\n\n"
    "Верни ТОЛЬКО JSON-объект без какого-либо текста до или после него:\n"
    '{"answer_relevance": "<high|medium|low>", '
    '"context_relevance": "<high|medium|low>", '
    '"answer_faithfulness": "<high|medium|low>", '
    '"overall_verdict": "<ok|retrieval_problem|generation_problem|hallucination|refusal_ok|mixed>", '
    '"reason": "<обоснование на русском, 1–2 предложения>"}\n\n'
    "Смысл overall_verdict:\n"
    '  "ok"                — ответ релевантен, контекст релевантен, факты не выдуманы.\n'
    '  "retrieval_problem" — найденный контекст плохо подходит к вопросу.\n'
    '  "generation_problem" — контекст в целом подходит, но ответ слабый, неполный или не по существу.\n'
    '  "hallucination"     — ответ содержит факты, которых нет в контексте.\n'
    '  "refusal_ok"        — отказ корректен: контекст не даёт ответа.\n'
    '  "mixed"             — смешанный случай, который не сводится к одной простой категории.'
)


# ====================================================================
#  ОЦЕНКА ПО РАЗМЕТКЕ
# ====================================================================

def classify(question: dict, hits, result: dict) -> str:
    """Классифицирует результат по разметке; порядок правил существенен."""
    text = result["answer"].strip()
    refused = text == REFUSAL_TEXT
    if not question["in_corpus"]:
        return "ok" if refused else "hallucination"
    if refused:
        return "over_refusal"
    if not set(question["expected"]).intersection(result["sources"]):
        return "retrieval_error"
    if question["must_contain"].casefold() not in text.casefold():
        return "generation_error"
    return "ok"


# ====================================================================
#  LLM-КАК-СУДЬЯ
# ====================================================================

def _parse_judgment(raw: str) -> dict:
    if raw.startswith("```") and raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    judgment = json.loads(raw)
    if not isinstance(judgment, dict):
        raise ValueError("Нужен JSON-объект")
    for field in ("answer_relevance", "context_relevance", "answer_faithfulness"):
        if judgment.get(field) not in {"high", "medium", "low"}:
            raise ValueError(f"{field} должен быть high, medium или low; получено {judgment.get(field)!r}")
    if judgment.get("overall_verdict") not in {
        "ok", "retrieval_problem", "generation_problem", "hallucination", "refusal_ok", "mixed"
    }:
        raise ValueError("Некорректный общий вердикт судьи")
    if not isinstance(judgment.get("reason"), str) or not judgment["reason"].strip():
        raise ValueError("Судья не указал пояснение")
    return judgment


def llm_judge(question: str, context: str, answer_text: str) -> dict:
    """Отдельный вызов GigaChat-судьи с проверкой формата и ограниченным повтором."""
    raw = ""
    try:
        user_msg = (f"ВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{context}"
                    f"\n\nОТВЕТ:\n{answer_text}\n\n"
                    "Оцени корректный отказ как отказ, а не как выдуманный факт. "
                    "При обоснованном отказе answer_faithfulness=high. "
                    "Значение refusal_ok может быть только в overall_verdict, "
                    "а во всех трёх критериях допустимы только high, medium, low.")
        messages = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg}]
        with GigaChat(
            credentials=(os.getenv("GIGACHAT_CREDENTIALS") or "").strip().strip("\"'"),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            verify_ssl_certs=False, max_retries=2,
        ) as client:
            for attempt in range(3):
                response = _retry_transport(lambda: client.chat({
                    "model": "GigaChat-2", "temperature": 0.0, "max_tokens": 700,
                    "messages": messages,
                }))
                raw = response.choices[0].message.content.strip()
                try:
                    return _parse_judgment(raw)
                except (ValueError, TypeError) as error:
                    if attempt == 2:
                        raise
                    messages.extend([
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": (
                            f"Исправь формат оценки: {error}. Верни весь JSON заново. "
                            "Во всех трёх полях answer_relevance, context_relevance, "
                            "answer_faithfulness используй ТОЛЬКО high, medium или low, "
                            "в том числе при отказе. refusal_ok допускается только в overall_verdict. "
                            "Не меняй оцениваемый вопрос, контекст и ответ."
                        )},
                    ])
    except Exception as error:
        return {"overall_verdict": "error", "reason": str(error), "raw_response": raw}


def _retry_transport(call):
    """Ограниченные повторы сетевых ошибок, HTTP 429 и временных ошибок API."""
    for attempt in range(5):
        try:
            return call()
        except (httpx.TransportError, RateLimitError, ServerError):
            if attempt == 4:
                raise
            time.sleep(min(5 * (attempt + 1), 20))


# ====================================================================
#  КОНТРОЛЬНЫЙ ПРОГОН
# ====================================================================

def run_eval(client, model, alias: str, verbose=True) -> dict:
    """Прогоняет контрольный набор и выводит два отдельных отчёта."""
    rows = []

    for q in QUESTIONS:
        print(f"Оцениваю {q['id']}...", flush=True)
        time.sleep(1)
        hits = search(client, model, q["question"], alias)
        result = _retry_transport(lambda: answer(client, model, q["question"], alias))
        ctx = format_context(hits) if hits else "(ничего не найдено)"

        rows.append({
            "id":        q.get("id", ""),
            "question":  q["question"],
            "in_corpus": q["in_corpus"],
            "result": result,
            "context": ctx,
            "markup":    classify(q, hits, result),
            "judgment":  llm_judge(q["question"], ctx, result["answer"]),
        })

    result_dir = BASE_DIR / "results" / "lesson8"
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    report = result_dir / f"eval_{manifest['chunk_size']}_{manifest['overlap']}.json"
    report.write_text(json.dumps({"manifest": manifest, "rows": rows}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    if verbose:
        _print_markup_report(rows)
        _print_judge_report(rows)

    # агрегаты для sweep_chunking
    counts = {"markup": {}, "judge": {}}
    for r in rows:
        mk = r["markup"]
        jv = r["judgment"].get("overall_verdict", "error")
        counts["markup"][mk] = counts["markup"].get(mk, 0) + 1
        counts["judge"][jv]  = counts["judge"].get(jv, 0) + 1
    counts["_rows"] = rows
    return counts


# ---------------------------------------------------------------------------
#  Метки для человекочитаемого вывода
# ---------------------------------------------------------------------------

_MARKUP_LABEL = {
    "ok":               "✓ верно",
    "hallucination":    "✗ галлюцинация",
    "retrieval_error":  "✗ документ не найден",
    "generation_error": "⚠ ответ неполный",
    "over_refusal":     "⚠ отказал зря",
}

_JUDGE_VERDICT_LABEL = {
    "ok":                   "всё хорошо",
    "retrieval_problem":    "контекст не подходит",
    "generation_problem":   "ответ слабый",
    "hallucination":        "галлюцинация",
    "refusal_ok":           "отказ обоснован",
    "mixed":                "смешанный случай",
    "error":                "ошибка вызова судьи",
}

_LEVEL_LABEL = {
    "high":   "✓ высокая",
    "medium": "⚠ средняя",
    "low":    "✗ низкая",
}


def _short(text: str, width: int) -> str:
    return text[:width - 1] + "…" if len(text) > width else text


def _print_markup_report(rows: list) -> None:
    print(_SEP2)
    print("ОТЧЁТ 1 — ОЦЕНКА ПО РАЗМЕТКЕ EVAL-НАБОРА")
    print(_SEP2)
    print()

    counts = {}
    for r in rows:
        mk = r["markup"]
        label = _MARKUP_LABEL.get(mk, mk)
        tag = "(вне базы)" if not r["in_corpus"] else ""
        qid = r["id"] or "—"
        print(f"  {qid:<4}  {label:<25}  {_short(r['question'], 55)}  {tag}")
        counts[mk] = counts.get(mk, 0) + 1

    print()
    print("  Итого:")
    order = ["ok", "retrieval_error", "generation_error", "hallucination", "over_refusal"]
    for mk in order:
        n = counts.get(mk, 0)
        if n or mk == "ok":
            print(f"    {_MARKUP_LABEL.get(mk, mk):<25}  {n}")
    print()


def _print_judge_report(rows: list) -> None:
    print(_SEP2)
    print("ОТЧЁТ 2 — ОЦЕНКА GigaChat-СУДЬЁЙ (LLM-as-Judge)")
    print(_SEP2)

    counts_verdict = {}
    for r in rows:
        j = r["judgment"]
        verdict = j.get("overall_verdict", "error")
        counts_verdict[verdict] = counts_verdict.get(verdict, 0) + 1

        def _level(val: str) -> str:
            # модель иногда возвращает нестандартное значение (напр. "refusal_ok")
            # нормализуем: берём только high/medium/low, остальное → "—"
            v = (val or "").lower().strip()
            if v in _LEVEL_LABEL:
                return _LEVEL_LABEL[v]
            for key in _LEVEL_LABEL:
                if v.startswith(key):
                    return _LEVEL_LABEL[key]
            return f"({val})"

        ar = _level(j.get("answer_relevance",    ""))
        cr = _level(j.get("context_relevance",   ""))
        af = _level(j.get("answer_faithfulness", ""))
        vl = _JUDGE_VERDICT_LABEL.get(verdict, verdict)
        reason = j.get("reason", "")
        tag = "(вне базы)" if not r["in_corpus"] else ""
        qid = r["id"] or "—"

        print()
        print(f"  {qid}  {_short(r['question'], 65)}  {tag}")
        print(f"      Ответ отвечает на вопрос:    {ar}")
        print(f"      Контекст подходит к вопросу: {cr}")
        print(f"      Ответ не выдумывает факты:   {af}")
        print(f"      Итог судьи: {vl}")
        print(f"      Пояснение: «{reason}»")

    print()
    print(_SEP)
    print("  Итого по судье:")
    order = ["ok", "refusal_ok", "generation_problem", "retrieval_problem",
             "hallucination", "mixed", "error"]
    for v in order:
        n = counts_verdict.get(v, 0)
        if n:
            print(f"    {_JUDGE_VERDICT_LABEL.get(v, v):<25}  {n}")
    print()


# ====================================================================
#  ПЕРЕБОР CHUNK_SIZE / OVERLAP
# ====================================================================

def sweep_chunking(client, model, alias: str):
    """Меряет все комбинации на неизменном наборе и восстанавливает 600/150."""
    total = sum(q["in_corpus"] for q in QUESTIONS)
    table = []
    print("Размер / нахлёст | чанков | по разметке | по судье", flush=True)
    try:
        for size, overlap in CHUNK_COMBOS:
            ingestion.CHUNK_SIZE, ingestion.OVERLAP = size, overlap
            manifest, _ = ingestion.build_index(client, model)
            counts = run_eval(client, model, alias, verbose=False)
            rows = [row for row in counts["_rows"] if row["in_corpus"]]
            markup = sum(row["markup"] == "ok" for row in rows)
            judge = sum(row["judgment"]["overall_verdict"] == "ok" for row in rows)
            errors = sum(row["judgment"]["overall_verdict"] == "error" for row in counts["_rows"])
            table.append({"size": size, "overlap": overlap, "chunks": manifest["chunks"],
                          "markup_ok": markup, "judge_ok": judge, "total": total, "judge_errors": errors})
            print(f"{size:4} / {overlap:<3} | {manifest['chunks']:6} | {markup}/{total} | {judge}/{total} (ошибок судьи: {errors})", flush=True)
    finally:
        ingestion.CHUNK_SIZE, ingestion.OVERLAP = 600, 150
        ingestion.build_index(client, model)
        print("Индекс восстановлен: 600/150", flush=True)
    print("\nИТОГОВАЯ ТАБЛИЦА")
    for row in table:
        print(f"{row['size']} / {row['overlap']} | {row['chunks']} | {row['markup_ok']}/{total} | {row['judge_ok']}/{total}")
    path = BASE_DIR / "results" / "lesson8" / "sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return table


# ====================================================================
#  КОНФЛИКТ ВЕРСИЙ ОДНОГО РЕГЛАМЕНТА
# ====================================================================

def resolve_conflicts(hits):
    """Удаляет старые редакции, сохраняя порядок и чанки актуальной версии."""
    groups = {}
    for hit in hits:
        key = hit.payload["doc_id"].removesuffix("-OLD")
        groups.setdefault(key, []).append(hit)
    keep = set()
    for group in groups.values():
        editions = {(hit.payload["doc_id"], hit.payload.get("date") or "") for hit in group}
        if len(editions) == 1:
            keep.update(id(hit) for hit in group)
            continue

        def edition(hit):
            raw_date = hit.payload.get("date") or ""
            date = datetime.min
            if raw_date:
                for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
                    try:
                        date = datetime.strptime(raw_date, fmt)
                        break
                    except ValueError:
                        pass
                else:
                    raise ValueError(f"Неизвестный формат даты: {raw_date}")
            return date, not hit.payload["doc_id"].endswith("-OLD")

        latest = max(group, key=edition)
        keep.update(id(hit) for hit in group
                    if hit.payload["doc_id"] == latest.payload["doc_id"]
                    and edition(hit) == edition(latest))
    return [hit for hit in hits if id(hit) in keep]


def demo_conflict(client, model, alias: str):
    """Показывает конфликт; фикстура и индекс восстанавливаются даже при ошибке."""
    fixture = CONFLICT_DIR / "REGL-GEN-01-OLD.json"
    target = CORPUS_DIR / fixture.name
    if target.exists():
        raise FileExistsError(f"В корпусе уже есть {target.name}; исходный файл не изменён")
    try:
        shutil.copy(fixture, target)
        manifest, _ = ingestion.build_index(client, model)
        print(f"КОНФЛИКТ ВЕРСИЙ: {manifest['chunks']} чанка")
        question = "В какой срок должны устраняться критичные неисправности насосного оборудования?"
        hits = search(client, model, question, alias, doc_type="регламент", limit=5)
        for label, selected in (("До resolve_conflicts", hits),
                                ("После resolve_conflicts", resolve_conflicts(hits))):
            print(label)
            for hit in selected:
                p = hit.payload
                print(f"{hit.score:.3f} {p['doc_id']} версия={p.get('version')} дата={p.get('date')} {p.get('section')}")
        filtered = resolve_conflicts(hits)
        assert any(h.payload['doc_id'] == 'REGL-GEN-01-OLD' for h in hits)
        assert any(h.payload['doc_id'] == 'REGL-GEN-01' for h in filtered)
        assert not any(h.payload['doc_id'] == 'REGL-GEN-01-OLD' for h in filtered)
    finally:
        target.unlink(missing_ok=True)
        manifest, _ = ingestion.build_index(client, model)
        print(f"Фикстура убрана; индекс восстановлен: {manifest['chunks']} чанков")


# ====================================================================
#  ОСНОВНОЙ ПРОГОН
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Оценка качества и защита RAG")
    parser.add_argument("--embedded", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="перебор CHUNK_SIZE/OVERLAP (пересобирает индекс)")
    parser.add_argument("--conflict", action="store_true",
                        help="демо конфликта версий регламента")
    args = parser.parse_args()

    manifest = load_manifest()
    client = make_client(args.embedded, DB_PATH)
    print("Загружаю модель...")
    model = SentenceTransformer(MODEL_NAME)
    alias = manifest["alias"]

    try:
        if args.sweep:
            sweep_chunking(client, model, alias)
        elif args.conflict:
            demo_conflict(client, model, alias)
        else:
            print(_SEP2)
            print("КОНТРОЛЬНЫЙ ПРОГОН: разметка + LLM-as-Judge (GigaChat)")
            print(_SEP2)
            run_eval(client, model, alias)
    finally:
        client.close()


if __name__ == "__main__":
    main()
