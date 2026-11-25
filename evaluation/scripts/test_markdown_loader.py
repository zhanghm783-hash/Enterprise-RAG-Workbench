"""验证标题分段不会拆开围栏代码块；纯本地测试，不调用外部服务。"""
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.rag.loader import _split_markdown_sections, build_loaded_document_from_text  # noqa: E402


class MarkdownSectionTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline test network call'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def titles(self, text):
        return [section.metadata.get('section_title') for section in _split_markdown_sections(text)]

    def test_slow_log_remains_in_one_section(self):
        text = '## 慢日志\n\n```log\n# Query_time: 0.18085\n# Cop_process: Avg_time: 170ms\n```\n'
        sections = _split_markdown_sections(text)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].metadata['section_title'], '慢日志')
        self.assertIn('# Query_time: 0.18085\n# Cop_process: Avg_time: 170ms', sections[0].text)
        self.assertIn('```log', sections[0].text)
        self.assertTrue(sections[0].text.endswith('```'))

    def test_normal_headings_and_metadata_still_split(self):
        sections = _split_markdown_sections('介绍\n# 概览\n正文\n## 操作\n步骤')
        self.assertEqual([s.metadata['section_title'] for s in sections], ['Introduction', '概览', '操作'])
        self.assertEqual([s.metadata['section_index'] for s in sections], [0, 1, 2])
        self.assertTrue(all(s.metadata['section_type'] == 'markdown_heading' for s in sections))

    def test_normal_heading_after_closed_code_block_is_recognized(self):
        text = '# 配置\n```toml\n# capacity\ncapacity=1\n```\n## 下一节\n正文'
        self.assertEqual(self.titles(text), ['配置', '下一节'])

    def test_tilde_fence_is_supported(self):
        self.assertEqual(self.titles('# 配置\n~~~toml\n# comment\n~~~\n## 说明\n正文'), ['配置', '说明'])

    def test_shorter_fence_cannot_close_longer_fence(self):
        text = '# 示例\n````markdown\n```\n# still code\n````\n## 后续'
        self.assertEqual(self.titles(text), ['示例', '后续'])

    def test_other_fence_character_cannot_close_block(self):
        text = '# 示例\n```log\n~~~\n# still code\n```\n## 后续'
        self.assertEqual(self.titles(text), ['示例', '后续'])

    def test_closing_fence_may_be_longer_and_have_whitespace(self):
        text = '# 示例\n```log\n# line\n  ````` \t\n## 后续'
        self.assertEqual(self.titles(text), ['示例', '后续'])

    def test_closing_fence_with_non_whitespace_is_not_a_close(self):
        text = '# 示例\n```log\n```oops\n# still code\n```\n## 后续'
        self.assertEqual(self.titles(text), ['示例', '后续'])

    def test_unclosed_fence_keeps_remaining_text_in_code_section(self):
        text = '# 示例\n```log\n# first\n## second'
        self.assertEqual(self.titles(text), ['示例'])

    def test_up_to_three_spaces_before_fence(self):
        text = '  # 示例\n   ```log\n# line\n   ```\n  ## 后续'
        self.assertEqual(self.titles(text), ['示例', '后续'])

    def test_invalid_atx_markers_do_not_create_headings(self):
        text = '# 实际标题\n#include <stdio.h>\n####### seven hashes\n    # indented code'
        self.assertEqual(self.titles(text), ['实际标题'])

    def test_backtick_in_opening_info_invalidates_backtick_fence(self):
        text = '# 示例\n```python `invalid`\n## 真标题'
        self.assertEqual(self.titles(text), ['示例', '真标题'])

    def test_tilde_opening_info_can_contain_backticks(self):
        text = '# 示例\n~~~python `allowed`\n# code\n~~~\n## 后续'
        self.assertEqual(self.titles(text), ['示例', '后续'])

    def test_inline_markdown_upload_uses_same_code_protection(self):
        doc = build_loaded_document_from_text('log.md', '# 日志\n```log\n# Query_time: 1\n```')
        self.assertEqual(len(doc.sections), 1)
        self.assertEqual(doc.filename, 'log.md')

    def test_ordinary_text_and_empty_input(self):
        self.assertEqual(self.titles('普通文本'), ['Introduction'])
        sections = _split_markdown_sections('')
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].text, '')


if __name__ == '__main__':
    unittest.main()
