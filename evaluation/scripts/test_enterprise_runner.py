"""验证实际候选执行器与跨块主指标的连接，不调用模型。"""
import copy
import unittest
from run_enterprise_retrieval import score_record,summarize


class EnterpriseRunnerTests(unittest.TestCase):
    def setUp(self):
        self.mapping={'required_unit_gold_chunk_ids':{'limit':['a'],'action':['b']},
                      'optional_unit_gold_chunk_ids':{'example':['e']},
                      'single_chunk_full_evidence_ids':[],'missing_required_units':[]}

    def record(self,ids,status='ok'):
        return {'results':{mode:{'hits':[{'chunk_id':i} for i in ids],'status':status,'hit_at_5':bool(ids)}
                           for mode in ('vector','bm25','hybrid')}}

    def test_complete_cross_chunk_evidence(self):
        result=score_record(self.record(['a','b']),self.mapping)
        self.assertTrue(result['results']['hybrid']['evidence_score']['complete_evidence_at_k'])
        self.assertNotIn('hit_at_5',result['results']['hybrid'])

    def test_optional_only_not_a_required_hit(self):
        result=score_record(self.record(['e']),self.mapping)
        self.assertFalse(result['results']['vector']['evidence_score']['any_evidence_hit_at_k'])

    def test_error_and_partial_remain_in_denominator(self):
        records=[score_record(self.record(['a','b']),self.mapping),score_record(self.record(['a']),self.mapping),
                 score_record(self.record([],'error'),self.mapping)]
        summary=summarize(records)['bm25']
        self.assertEqual(summary['questions'],3)
        self.assertEqual(summary['complete_evidence_at_5'],1/3)
        self.assertEqual(summary['any_evidence_hit_at_5'],2/3)
        self.assertEqual(summary['statuses']['error'],1)

    def test_missing_mapping_rejected(self):
        mapping=copy.deepcopy(self.mapping);mapping['missing_required_units']=['action']
        with self.assertRaises(ValueError):score_record(self.record(['a']),mapping)


if __name__=='__main__':unittest.main()
