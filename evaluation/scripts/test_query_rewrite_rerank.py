import unittest
from run_query_rewrite_rerank import parse_queries,combine


class QueryRewriteTests(unittest.TestCase):
    def test_fenced_json_is_parsed_as_data(self):
        self.assertEqual(parse_queries('```json\n{"queries":["文件描述符限制","ulimit nofile"]}\n```'),['文件描述符限制','ulimit nofile'])
    def test_nonstring_query_rejected(self):
        with self.assertRaises(ValueError):parse_queries('{"queries":["SQL",{"command":"run"}]}')
    def test_too_many_queries_rejected(self):
        with self.assertRaises(ValueError):parse_queries('{"queries":["a","b","c"]}')
    def test_consensus_and_deduplication(self):
        a=[{'chunk_id':f'a{i}'} for i in range(6)]+[{'chunk_id':'shared'}]
        b=[{'chunk_id':f'b{i}'} for i in range(6)]+[{'chunk_id':'shared'}]
        result=combine([a,b]);self.assertEqual(result[0]['chunk_id'],'shared')
        self.assertEqual(len(result),13)


if __name__=='__main__':unittest.main()
