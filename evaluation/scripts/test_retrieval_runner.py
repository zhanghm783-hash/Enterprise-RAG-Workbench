"""用虚拟候选检查对照实验的隔离和计分，禁止真实网络访问。"""
import copy
import logging
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import run_retrieval as runner
from app.rag import retriever


class RetrievalRunnerTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline test network call'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.expected = {str(i): {'id': str(i), 'content': f'body {i}', 'document_id': 'document'}
                         for i in range(1, 7)}
        self.question = {'id': 'synthetic', 'question': 'query only', 'topic': 'synthetic', 'kind': 'synthetic',
                         'reference_answer': 'not sent to retriever', 'evidence_options': [],
                         'baseline_full_evidence_chunks': [{'chunk_id': '2'}]}

    def hit(self, index, source='vector'):
        return {'chunk_id': str(index), 'document_id': 'document', 'content': f'body {index}',
                'filename': 'synthetic.md', 'score': 1 / index, f'{source}_score': 1 / index,
                'retrieval_source': source, 'retrieval_sources': [source]}

    def run_case(self, modes, vector=None, bm25=None):
        return runner.run_question(self.question, modes=modes, expected=self.expected,
                                   vector_fn=vector if vector is not None else Mock(return_value=[]),
                                   bm25_fn=bm25 if bm25 is not None else Mock(return_value=[]))

    def test_vector_mode_never_calls_bm25_or_fallback(self):
        vector, bm25 = Mock(return_value=[self.hit(2)]), Mock(side_effect=AssertionError('BM25 called'))
        with patch.object(retriever, '_postgres_fallback', side_effect=AssertionError('Fallback called')):
            result = self.run_case(['vector'], vector, bm25)
        self.assertTrue(result['results']['vector']['hit_at_5'])
        vector.assert_called_once_with(query='query only', candidate_k=20, raise_on_error=True)
        bm25.assert_not_called()

    def test_bm25_mode_never_calls_vector(self):
        vector, bm25 = Mock(side_effect=AssertionError('Vector called')), Mock(return_value=[self.hit(2, 'bm25')])
        self.run_case(['bm25'], vector, bm25)
        vector.assert_not_called()
        bm25.assert_called_once_with(query='query only', candidate_k=20, raise_on_error=True)

    def test_all_modes_reuse_candidates_and_rrf_promotes_shared_hit(self):
        original = [self.hit(1), self.hit(2)]
        before = copy.deepcopy(original)
        vector, bm25 = Mock(return_value=original), Mock(return_value=[self.hit(2, 'bm25')])
        result = self.run_case(list(runner.MODES), vector, bm25)
        vector.assert_called_once()
        bm25.assert_called_once()
        self.assertEqual(result['results']['vector']['hits'][0]['chunk_id'], '1')
        self.assertEqual(result['results']['hybrid']['hits'][0]['chunk_id'], '2')
        self.assertAlmostEqual(result['results']['hybrid']['hits'][0]['fused_score'], 1 / 62 + 1 / 61)
        self.assertEqual(original, before)

    def test_rank_six_does_not_count_as_top_five_hit(self):
        self.question['baseline_full_evidence_chunks'] = [{'chunk_id': '6'}]
        result = self.run_case(['vector'], Mock(return_value=[self.hit(i) for i in range(1, 7)]))
        self.assertFalse(result['results']['vector']['hit_at_5'])
        self.assertEqual(result['results']['vector']['returned_count'], 5)

    def test_empty_result_is_successful_execution_but_miss(self):
        with patch.object(retriever, '_postgres_fallback', side_effect=AssertionError('Fallback called')):
            result = self.run_case(list(runner.MODES))
        for entry in result['results'].values():
            self.assertEqual(entry['status'], 'ok')
            self.assertFalse(entry['hit_at_5'])

    def test_error_is_retained_and_hybrid_is_marked_degraded(self):
        result = self.run_case(list(runner.MODES), Mock(side_effect=ConnectionError('sensitive detail')),
                               Mock(return_value=[self.hit(2, 'bm25')]))
        self.assertEqual(result['results']['vector']['status'], 'error')
        self.assertEqual(result['results']['hybrid']['status'], 'degraded')
        self.assertNotIn('sensitive detail', str(result))
        summary = runner.summarize([result], list(runner.MODES))
        self.assertEqual(summary['vector']['questions'], 1)
        self.assertEqual(summary['vector']['hit_rate_at_5_including_execution_failures'], 0)

    def test_stale_or_foreign_candidate_is_execution_error(self):
        hit = self.hit(2)
        hit['content'] = 'changed outside snapshot'
        result = self.run_case(['vector'], Mock(return_value=[hit]))
        self.assertEqual(result['results']['vector']['status'], 'error')
        self.assertFalse(result['results']['vector']['hit_at_5'])

    def test_real_helpers_raise_in_strict_mode_and_keep_default_fallback_behavior(self):
        for function, target in ((retriever._vector_retrieve_chunks, 'get_vector_store'),
                                 (retriever._bm25_retrieve_chunks, 'get_bm25_index')):
            with self.subTest(target=target), patch.object(retriever, target, side_effect=ConnectionError('offline')):
                with self.assertRaises(ConnectionError):
                    function(query='query', candidate_k=20, raise_on_error=True)
                self.assertEqual(function(query='query', candidate_k=20), [])

    def test_embedding_requires_explicit_flag_before_loading_data(self):
        with patch.object(runner, 'get_settings', side_effect=AssertionError('Settings loaded')):
            with self.assertRaises(SystemExit) as result:
                runner.main(['--modes', 'vector'])
        self.assertEqual(result.exception.code, 2)

    def test_draft_formal_set_cannot_run(self):
        with patch.object(runner, 'get_settings', side_effect=AssertionError('Settings loaded')):
            with self.assertRaises(SystemExit) as result:
                runner.main(['--modes', 'bm25', '--split', 'test'])
        self.assertEqual(result.exception.code, 2)

    def test_usage_recorder_preserves_response_and_drops_sensitive_fields(self):
        response = SimpleNamespace(status_code=200, request_id='synthetic-request',
                                   usage={'total_tokens': 7, 'secret_field': 'do not export'})
        original = Mock(return_value=response)
        events = []
        tracked = runner.track_embedding_call(original, events)
        self.assertIs(tracked(input='query', model='test-model', api_key='do not export'), response)
        self.assertEqual(events[0]['usage'], {'total_tokens': 7})
        self.assertEqual(events[0]['request_id'], 'synthetic-request')
        self.assertNotIn('do not export', str(events))

    def test_usage_recorder_keeps_exception_type_without_message(self):
        events = []
        tracked = runner.track_embedding_call(Mock(side_effect=ConnectionError('do not export')), events)
        with self.assertRaises(ConnectionError):
            tracked(model='test-model')
        self.assertEqual(events[0]['error_type'], 'ConnectionError')
        self.assertNotIn('do not export', str(events))

    def test_frozen_question_hash_mismatch_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = runner.Path(folder) / 'questions.json'
            dataset.write_text('{"status":"frozen"}', encoding='utf-8')
            dataset.with_suffix('.sha256').write_text('0' * 64, encoding='ascii')
            with patch.object(runner, 'get_settings', side_effect=AssertionError('Settings loaded')):
                with self.assertRaisesRegex(ValueError, 'Frozen question file changed'):
                    runner.main(['--modes', 'bm25', '--split', 'test', '--dataset', str(dataset)])


if __name__ == '__main__':
    unittest.main()
