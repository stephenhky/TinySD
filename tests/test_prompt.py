import unittest

from tinysd.prompt import _convert_a1111_segment_to_compel, convert_a1111_to_compel


class TestConvertSegment(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(_convert_a1111_segment_to_compel("a cat"), "a cat")

    def test_lora_tag_removed(self):
        self.assertEqual(_convert_a1111_segment_to_compel("a cat <lora:foo:0.8>"), "a cat ")

    def test_weighted_parentheses(self):
        self.assertEqual(_convert_a1111_segment_to_compel("(cat:1.3)"), "(cat)1.3")

    def test_weighted_parentheses_with_spaces(self):
        self.assertEqual(_convert_a1111_segment_to_compel("(cat: 0.5 )"), "(cat)0.5")

    def test_weighted_brackets(self):
        self.assertEqual(_convert_a1111_segment_to_compel("[cat:0.7]"), "(cat)0.7")

    def test_bare_brackets_deemphasis(self):
        self.assertEqual(_convert_a1111_segment_to_compel("[cat]"), "(cat)-")

    def test_nested_brackets(self):
        self.assertEqual(_convert_a1111_segment_to_compel("[[cat]]"), "((cat)-)-")

    def test_bare_parentheses_emphasis(self):
        self.assertEqual(_convert_a1111_segment_to_compel("(cat)"), "(cat)+")

    def test_nested_bare_parentheses(self):
        self.assertEqual(_convert_a1111_segment_to_compel("((cat))"), "((cat)+)+")

    def test_nested_weighted_inside_bare(self):
        self.assertEqual(_convert_a1111_segment_to_compel("((cat:1.2))"), "((cat)1.2)+")

    def test_escaped_parentheses_untouched(self):
        self.assertEqual(_convert_a1111_segment_to_compel(r"\(cat\)"), r"\(cat\)")

    def test_existing_compel_suffix_kept(self):
        self.assertEqual(_convert_a1111_segment_to_compel("(cat)++"), "(cat)++")
        self.assertEqual(_convert_a1111_segment_to_compel("(cat)-"), "(cat)-")


class TestConvertA1111ToCompel(unittest.TestCase):
    def test_empty_and_whitespace_returned_as_is(self):
        self.assertEqual(convert_a1111_to_compel(""), "")
        self.assertEqual(convert_a1111_to_compel("   "), "   ")

    def test_none_returned_as_is(self):
        self.assertIsNone(convert_a1111_to_compel(None))

    def test_mixed_syntax(self):
        self.assertEqual(
            convert_a1111_to_compel("a (cat:1.3) on a [mat]"),
            "a (cat)1.3 on a (mat)-",
        )

    def test_result_is_stripped(self):
        self.assertEqual(convert_a1111_to_compel("  a cat <lora:x:1>  "), "a cat")

    def test_break_becomes_and(self):
        self.assertEqual(
            convert_a1111_to_compel("a cat BREAK a dog"),
            '("a cat", "a dog").and()',
        )

    def test_break_escapes_quotes(self):
        self.assertEqual(
            convert_a1111_to_compel('say "hi" BREAK dog'),
            '("say \\"hi\\"", "dog").and()',
        )

    def test_break_requires_word_boundary(self):
        self.assertEqual(convert_a1111_to_compel("BREAKFAST"), "BREAKFAST")

    def test_existing_compel_conjunction_untouched(self):
        prompt = '("a cat", "a dog").and()'
        self.assertEqual(convert_a1111_to_compel(prompt), prompt)
        prompt = '("a cat", "a dog").blend(0.3, 0.7)'
        self.assertEqual(convert_a1111_to_compel(prompt), prompt)


if __name__ == "__main__":
    unittest.main()
