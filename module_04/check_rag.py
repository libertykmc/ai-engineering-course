"""Локальные проверки контекста, промпта и отказа без обращения к API."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import rag_assistant as rag


def hit(score, text='Полный текст источника', page=None, section='5.1'):
    return SimpleNamespace(score=score, payload={
        'doc_id': 'REGL-GEN-01', 'page': page, 'section': section, 'text': text,
    })


class RagChecks(unittest.TestCase):
    def test_full_context_and_locations(self):
        text = 'Начало\n' + 'а' * 700 + '\nКлючевой факт в самом конце'
        chunks = [hit(0.8, text), hit(0.7, 'Страница PDF', page=24, section=None)]
        context = rag.format_context(chunks)
        self.assertIn('[1] Источник: REGL-GEN-01, раздел 5.1\n' + text, context)
        self.assertIn('\n\n[2] Источник: REGL-GEN-01, стр. 24\nСтраница PDF', context)
        self.assertEqual(rag.format_context([]), '')

    def test_prompt_contract(self):
        messages = rag.build_prompt('Мой вопрос?', 'Найденный текст')
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertIn(rag.REFUSAL_TEXT, messages[0]['content'])
        self.assertIn('[doc_id, место]', messages[0]['content'])
        self.assertIn('исключительно на основании', messages[0]['content'])
        self.assertEqual(messages[1]['content'], 'КОНТЕКСТ:\nНайденный текст\n\nВОПРОС:\nМой вопрос?')

    def test_gate_boundary(self):
        self.assertFalse(rag.confidence_gate([]))
        self.assertFalse(rag.confidence_gate([hit(rag.MIN_SCORE - 0.001)]))
        self.assertTrue(rag.confidence_gate([hit(rag.MIN_SCORE)]))
        self.assertTrue(rag.confidence_gate([hit(rag.MIN_SCORE + 0.001)]))

    def test_refusal_skips_generator(self):
        for hits in ([], [hit(0.1)]):
            with self.subTest(hits=hits), patch.object(rag, 'search', return_value=hits), patch.object(rag, 'ask_model') as generator:
                result = rag.answer(None, None, 'Вопрос вне корпуса', 'promtech')
                self.assertEqual(result['answer'], rag.REFUSAL_TEXT)
                self.assertEqual(result['sources'], [])
                self.assertEqual(result['refused_by'], 'score')
                generator.assert_not_called()

    def test_answer_passes_question_and_full_context(self):
        chunk = hit(0.8, 'Достоверный факт из документа')
        with patch.object(rag, 'search', return_value=[chunk]), patch.object(rag, 'ask_model', return_value='Ответ [REGL-GEN-01, раздел 5.1]') as generator:
            result = rag.answer(None, None, 'Мой вопрос?', 'promtech')
            messages = generator.call_args.args[0]
            self.assertIn(chunk.payload['text'], messages[1]['content'])
            self.assertIn('Мой вопрос?', messages[1]['content'])
            self.assertEqual(result['sources'], ['REGL-GEN-01'])
            self.assertIsNone(result['refused_by'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
