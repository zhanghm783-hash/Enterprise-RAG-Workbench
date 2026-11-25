"""跨块依据评分的边界测试，不连接数据库或模型。"""
import unittest

from enterprise_evidence_scoring import compile_mapping, score_hits


class EnterpriseEvidenceScoringTests(unittest.TestCase):
    def setUp(self):
        self.rule = {'id': 'Q', 'required_units': [
            {'id': 'limit', 'options': [{'filename': 'a.md', 'quotes': ['TTL', '1 hour']}]},
            {'id': 'action', 'options': [{'filename': 'a.md', 'quotes': ['split transaction']}]}],
            'optional_units': [{'id': 'example', 'options': [{'filename': 'a.md', 'quotes': ['numeric log']}]}]}
        self.chunks = [{'chunk_id': 'limit', 'corpus_path': 'a.md', 'content': 'TTL: 1 hour'},
                       {'chunk_id': 'action', 'corpus_path': 'a.md', 'content': 'Please split transaction'},
                       {'chunk_id': 'example', 'corpus_path': 'a.md', 'content': 'numeric log'}]
        self.mapping = compile_mapping(self.rule, self.chunks)

    def test_two_hits_can_complete_two_required_units(self):
        result = score_hits(self.mapping, ['limit', 'action'])
        self.assertTrue(result['complete_evidence_at_k'])
        self.assertFalse(result['single_chunk_complete_hit_at_k'])

    def test_partial_hit_is_not_complete(self):
        result = score_hits(self.mapping, ['limit'])
        self.assertTrue(result['any_evidence_hit_at_k'])
        self.assertFalse(result['complete_evidence_at_k'])
        self.assertEqual(result['evidence_unit_coverage_fraction'], 0.5)

    def test_optional_only_never_counts_as_required_hit(self):
        result = score_hits(self.mapping, ['example'])
        self.assertFalse(result['any_evidence_hit_at_k'])
        self.assertFalse(result['complete_evidence_at_k'])

    def test_optional_missing_does_not_prevent_completion(self):
        self.assertTrue(score_hits(self.mapping, ['action', 'limit'])['complete_evidence_at_k'])

    def test_rank_six_does_not_count(self):
        result = score_hits(self.mapping, ['limit', 'x1', 'x2', 'x3', 'x4', 'action'])
        self.assertFalse(result['complete_evidence_at_k'])

    def test_duplicates_do_not_fill_missing_unit_or_extend_top_k(self):
        self.assertFalse(score_hits(self.mapping, ['limit']*5+['action'])['complete_evidence_at_k'])

    def test_equivalent_option_and_source_isolation(self):
        self.rule['required_units'][0]['options'].append({'filename': 'b.md', 'quotes': ['one hour timeout']})
        chunks = self.chunks + [{'chunk_id': 'equivalent', 'corpus_path': 'b.md', 'content': 'one hour timeout'},
                               {'chunk_id': 'wrong-source', 'corpus_path': 'c.md', 'content': 'one hour timeout'}]
        mapping = compile_mapping(self.rule, chunks)
        self.assertTrue(score_hits(mapping, ['equivalent', 'action'])['complete_evidence_at_k'])
        self.assertFalse(score_hits(mapping, ['wrong-source', 'action'])['complete_evidence_at_k'])

    def test_one_chunk_can_cover_all_units(self):
        chunks = self.chunks + [{'chunk_id': 'both', 'corpus_path': 'a.md', 'content': 'TTL: 1 hour; split transaction'}]
        self.assertTrue(score_hits(compile_mapping(self.rule, chunks), ['both'])['single_chunk_complete_hit_at_k'])

    def test_unmapped_unit_blocks_scoring(self):
        with self.assertRaises(ValueError):
            score_hits(compile_mapping(self.rule, self.chunks[:1]), ['limit'])

    def test_empty_hits_are_misses(self):
        result = score_hits(self.mapping, [])
        self.assertFalse(result['complete_evidence_at_k'])
        self.assertFalse(result['any_evidence_hit_at_k'])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            compile_mapping(self.rule, self.chunks + self.chunks[:1])

    def test_invalid_k_rejected(self):
        with self.assertRaises(ValueError):
            score_hits(self.mapping, [], k=0)


if __name__ == '__main__':
    unittest.main()
