import unittest
from types import SimpleNamespace
from run_rerank_experiment import parse_rerank


class RerankResponseTests(unittest.TestCase):
    def setUp(self):self.candidates=[{'chunk_id':'a'},{'chunk_id':'b'}]
    def response(self,rows,status=200):return SimpleNamespace(status_code=status,output={'results':rows})
    def test_indices_preserve_identity(self):
        result=parse_rerank(self.response([{'index':1,'relevance_score':.9},{'index':0,'relevance_score':.2}]),self.candidates)
        self.assertEqual([r['chunk_id'] for r in result],['b','a'])
    def test_duplicate_indices_rejected(self):
        with self.assertRaises(ValueError):parse_rerank(self.response([{'index':0,'relevance_score':.9}]*2),self.candidates)
    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):parse_rerank(self.response([{'index':2,'relevance_score':.9}]),self.candidates)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):parse_rerank(self.response([{'index':0,'relevance_score':float('nan')}]),self.candidates)
    def test_api_error_rejected(self):
        with self.assertRaises(RuntimeError):parse_rerank(self.response([],401),self.candidates)


if __name__=='__main__':unittest.main()
