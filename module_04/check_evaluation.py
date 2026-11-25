"""Проверки правил оценки, контракта судьи и восстановления состояния без API."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import evaluate_rag as evaluation


def hit(doc_id, date, score=0.8):
    return SimpleNamespace(payload={'doc_id': doc_id, 'date': date}, score=score)


class EvaluationChecks(unittest.TestCase):
    def test_classification_and_priority(self):
        question = {'in_corpus': True, 'expected': ['DOC'], 'must_contain': 'Точный факт'}
        examples = [
            ({'answer': 'ТОЧНЫЙ ФАКТ [DOC]', 'sources': ['DOC']}, 'ok'),
            ({'answer': 'Точный факт', 'sources': ['OTHER']}, 'retrieval_error'),
            ({'answer': 'Пересказ', 'sources': ['DOC']}, 'generation_error'),
            ({'answer': evaluation.REFUSAL_TEXT, 'sources': []}, 'over_refusal'),
        ]
        for result, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(evaluation.classify(question, [], result), expected)
        outside = {**question, 'in_corpus': False}
        self.assertEqual(evaluation.classify(outside, [], {'answer': evaluation.REFUSAL_TEXT}), 'ok')
        self.assertEqual(evaluation.classify(outside, [], {'answer': 'Общие знания'}), 'hallucination')

    def test_judge_messages_and_valid_json(self):
        judgment = {'answer_relevance': 'high', 'context_relevance': 'medium',
                    'answer_faithfulness': 'high', 'overall_verdict': 'mixed', 'reason': 'Контекст частичный.'}
        mock = MagicMock()
        mock.__enter__.return_value.chat.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n'+json.dumps(judgment)+'\n```'))])
        with patch.object(evaluation, 'GigaChat', return_value=mock):
            self.assertEqual(evaluation.llm_judge('Вопрос', 'Контекст', 'Ответ'), judgment)
        messages = mock.__enter__.return_value.chat.call_args.args[0]['messages']
        self.assertEqual(messages[0]['content'], evaluation.JUDGE_SYSTEM)
        for value in ('Вопрос', 'Контекст', 'Ответ'):
            self.assertIn(value, messages[1]['content'])
        self.assertNotIn('must_contain', messages[1]['content'])

    def test_judge_failure_is_not_success(self):
        for content in ('not json', '[]', '{"overall_verdict":"ok"}'):
            mock = MagicMock()
            mock.__enter__.return_value.chat.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            with patch.object(evaluation, 'GigaChat', return_value=mock):
                result = evaluation.llm_judge('Вопрос', 'Контекст', 'Ответ')
                self.assertEqual(result['overall_verdict'], 'error')
                self.assertTrue(result['reason'])

    def test_conflict_uses_date_not_score_and_keeps_order(self):
        old = hit('REGL-OLD', '2022-05-14', 0.95)
        new = hit('REGL', '2024-01-10', 0.7)
        other = hit('OTHER', '', 0.8)
        self.assertEqual(evaluation.resolve_conflicts([old, other, new]), [other, new])
        self.assertEqual(new.score, 0.7)
        self.assertEqual(evaluation.resolve_conflicts([]), [])
        self.assertEqual(evaluation.resolve_conflicts([old]), [old])

    def test_current_chunks_and_unpaired_documents_survive(self):
        old = hit('REGL-OLD', '2022-05-14')
        chunks = [hit('REGL', '2024-01-10'), hit('REGL', '2024-01-10')]
        self.assertEqual(evaluation.resolve_conflicts([chunks[0], old, chunks[1]]), chunks)
        self.assertEqual(evaluation.resolve_conflicts(chunks), chunks)
        stale_same_id = hit('REGL', '2022-05-14')
        self.assertEqual(evaluation.resolve_conflicts([stale_same_id, *chunks]), chunks)

    def test_sweep_excludes_outside_questions(self):
        rows = [
            {'in_corpus': True, 'markup': 'ok', 'judgment': {'overall_verdict': 'ok'}},
            {'in_corpus': True, 'markup': 'generation_error', 'judgment': {'overall_verdict': 'ok'}},
            {'in_corpus': False, 'markup': 'ok', 'judgment': {'overall_verdict': 'ok'}},
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluation, 'BASE_DIR', Path(directory)), \
                patch.object(evaluation, 'QUESTIONS', [{'in_corpus': True}]*2+[{'in_corpus': False}]), \
                patch.object(evaluation, 'CHUNK_COMBOS', [(400, 0)]), \
                patch.object(evaluation.ingestion, 'build_index', return_value=({'chunks': 12}, [])) as build, \
                patch.object(evaluation, 'run_eval', return_value={'_rows': rows}):
            table = evaluation.sweep_chunking(None, None, 'alias')
            self.assertEqual(table[0]['markup_ok'], 1)
            self.assertEqual(table[0]['judge_ok'], 2)
            self.assertEqual(table[0]['total'], 2)
            self.assertEqual(build.call_count, 2)
        self.assertEqual((evaluation.ingestion.CHUNK_SIZE, evaluation.ingestion.OVERLAP), (600, 150))

    def test_sweep_restores_after_error(self):
        with patch.object(evaluation, 'CHUNK_COMBOS', [(400, 0)]), \
                patch.object(evaluation.ingestion, 'build_index', return_value=({'chunks': 12}, [])) as build, \
                patch.object(evaluation, 'run_eval', side_effect=RuntimeError('ошибка прогона')):
            with self.assertRaises(RuntimeError):
                evaluation.sweep_chunking(None, None, 'alias')
            self.assertEqual(build.call_count, 2)
        self.assertEqual((evaluation.ingestion.CHUNK_SIZE, evaluation.ingestion.OVERLAP), (600, 150))

    def test_conflict_fixture_is_removed_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'fixtures').mkdir()
            (root/'corpus').mkdir()
            (root/'fixtures/REGL-GEN-01-OLD.json').write_text('{}')
            with patch.object(evaluation, 'CONFLICT_DIR', root/'fixtures'), \
                    patch.object(evaluation, 'CORPUS_DIR', root/'corpus'), \
                    patch.object(evaluation.ingestion, 'build_index', return_value=({'chunks': 921}, [])) as build, \
                    patch.object(evaluation, 'search', side_effect=RuntimeError('сбой поиска')):
                with self.assertRaises(RuntimeError):
                    evaluation.demo_conflict(None, None, 'alias')
                self.assertFalse((root/'corpus/REGL-GEN-01-OLD.json').exists())
                self.assertEqual(build.call_count, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
