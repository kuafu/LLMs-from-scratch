import contextlib
import copy
import io
import json
import os
import tempfile
import unittest

import torch

from gpt2toy import GPTModel
from gpt2toy_data import SimpleTokenizerV2
from gpt2toy_diagnostics import (
    analyze_and_print_cross_step_insights,
    collect_same_token_example,
)
from run_gpt2toy import (
    REPEATED_CONTEXT,
    SEED,
    TOY_CONFIG,
    TOY_VOCAB,
    main,
    run,
)


def make_default_model_and_tokenizer(seed=SEED):
    """建立与教学入口一致的 eval 模型和微型分词器，便于独立数值验算。"""
    torch.manual_seed(seed)
    model = GPTModel(TOY_CONFIG).eval()
    return model, SimpleTokenizerV2(TOY_VOCAB)


def embedded_trace(html):
    """从 HTML 面板读取内嵌的剖析数据，检查导出过程是否保留真实模型信息。"""
    marker = "const TRACE = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    return json.loads(html[start:end])


class GPTToyTeachingTests(unittest.TestCase):
    """验证教学入口、独立重复词探针、比较结论及 Markdown/HTML 导出。"""

    def test_default_run_exports_exact_console_and_independent_repeat_example(self):
        """完整运行应生成正确容量的模型、独立探针和与终端一致的日志。"""
        with tempfile.TemporaryDirectory() as output_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                payload = run(output_dir)

            console = stdout.getvalue()
            with open(os.path.join(output_dir, "gpt2toy.txt"), encoding="utf-8") as stream:
                saved_console = stream.read()
            self.assertEqual(console, saved_console)

            md_path = os.path.join(output_dir, "gpt2toy_dissection.md")
            html_path = os.path.join(output_dir, "gpt2toy_inspector.html")
            self.assertTrue(os.path.isfile(md_path))
            self.assertTrue(os.path.isfile(html_path))
            with open(md_path, encoding="utf-8") as stream:
                markdown = stream.read()
            with open(html_path, encoding="utf-8") as stream:
                html = stream.read()
            html_payload = embedded_trace(html)

        self.assertEqual(payload["config"]["context_length"], 5)
        self.assertEqual(payload["param_stats"]["total_p"], 308)
        self.assertEqual(len(payload["weights"]["pos_emb_weight"]), 5)
        self.assertEqual(len(payload["weights"]["pos_emb_weight"][0]), 4)
        self.assertEqual(len(payload["weights"]["blocks"][0]["causal_mask"]), 5)
        self.assertEqual(len(payload["weights"]["blocks"][0]["causal_mask"][0]), 5)

        steps = payload["steps"]
        self.assertEqual([len(step["input_ids"]) for step in steps], [4, 5])
        self.assertEqual([step["full_seq_len"] for step in steps], [4, 5])
        self.assertTrue(all(step["training"] is False for step in steps))
        self.assertEqual(len(payload["final_ids"]), 6)
        self.assertEqual(steps[1]["input_ids"][-1], TOY_VOCAB["<|endoftext|>"])
        self.assertEqual(steps[1]["input_tokens"][-1], "<|endoftext|>")

        repeat = payload["comparisons"]["same_token_diff_pos"]
        self.assertTrue(repeat["available"], repeat.get("reason"))
        self.assertEqual(repeat["input_text"], REPEATED_CONTEXT)
        self.assertEqual(repeat["input_ids"], [2, 3, 4, 5, 4])
        self.assertEqual((repeat["pos_a"], repeat["pos_b"]), (2, 4))
        self.assertEqual((repeat["token_id"], repeat["token_str"]), (4, "I"))
        self.assertEqual(repeat["token_labels"][2], "t2[ID=4:'I']")
        self.assertEqual(repeat["token_labels"][4], "t4[ID=4:'I']")

        decoded = SimpleTokenizerV2(TOY_VOCAB).decode(payload["final_ids"])
        self.assertEqual(payload["final_text"], decoded)
        for generated in (step["prediction"]["next_token_str"] for step in steps):
            self.assertIn(generated, console)
            self.assertIn(generated, markdown)
        self.assertIn(payload["final_text"], console)
        normalized_markdown = markdown.replace("\\|", "|")
        self.assertIn(payload["final_text"], normalized_markdown)
        self.assertEqual(html_payload["final_text"], payload["final_text"])
        self.assertEqual(
            [step["prediction"]["next_token_str"] for step in html_payload["steps"]],
            [step["prediction"]["next_token_str"] for step in steps],
        )
        self.assertIn("t2[ID=4", markdown)
        self.assertIn("t4[ID=4", html_payload["comparisons"]["same_token_diff_pos"]["token_labels"][4])

    def test_fixed_repeat_snapshot_matches_a_direct_quiet_forward_and_argmax_labels(self):
        """固定重复词快照及 Argmax 预测必须来自该输入的真实前向计算。"""
        model, tokenizer = make_default_model_and_tokenizer()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            example = collect_same_token_example(model, tokenizer, REPEATED_CONTEXT)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(example["available"], example.get("reason"))

        ids = torch.tensor([example["input_ids"]], dtype=torch.long)
        with torch.no_grad():
            direct_logits = model(ids)[0]
        recorded_logits = torch.tensor(example["trace"]["output_stage"]["logits_all"])
        torch.testing.assert_close(recorded_logits, direct_logits)

        self.assertEqual(example["trace"]["training"], False)
        self.assertTrue(example["token_embeddings_equal"])
        self.assertFalse(example["position_embeddings_equal"])
        self.assertFalse(example["logits_equal"])
        for field, position in (("prediction_a", example["pos_a"]), ("prediction_b", example["pos_b"])):
            token_id = int(direct_logits[position].argmax().item())
            self.assertEqual(example[field]["id"], token_id)
            self.assertEqual(example[field]["token"], tokenizer.int_to_str[token_id])

    def test_comparison_reports_verified_different_and_not_comparable_truthfully(self):
        """数值超差、训练模式或窗口移位时，不得误报 KV Cache 前缀验证成功。"""
        with tempfile.TemporaryDirectory() as output_dir, contextlib.redirect_stdout(io.StringIO()):
            payload = main(output_dir)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            verified = analyze_and_print_cross_step_insights(payload["steps"])
        self.assertEqual(verified["invariance"]["status"], "verified")
        self.assertTrue(verified["invariance_checks"])
        self.assertTrue(all(check["passed"] for check in verified["invariance_checks"]))
        self.assertEqual(verified["same_token_diff_pos"]["available"], False)

        changed_steps = copy.deepcopy(payload["steps"])
        changed_steps[1]["output_stage"]["logits_all"][0][0] += 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            changed = analyze_and_print_cross_step_insights(changed_steps)
        self.assertEqual(changed["invariance"]["status"], "different")
        self.assertTrue(any(not check["passed"] for check in changed["invariance_checks"]))

        training_steps = copy.deepcopy(payload["steps"])
        training_steps[1]["training"] = True
        with contextlib.redirect_stdout(io.StringIO()):
            training_mismatch = analyze_and_print_cross_step_insights(training_steps)
        self.assertEqual(training_mismatch["invariance"]["status"], "not_comparable")

        cropped_context = "Hello, I am Hello, I am"
        with tempfile.TemporaryDirectory() as output_dir, contextlib.redirect_stdout(io.StringIO()):
            cropped_payload = main(output_dir, start_context=cropped_context)
        with contextlib.redirect_stdout(io.StringIO()):
            cropped = analyze_and_print_cross_step_insights(cropped_payload["steps"])
        self.assertEqual(cropped["invariance"]["status"], "not_comparable")
        self.assertNotEqual(cropped["invariance"]["reason"], "")

    def test_alternate_configuration_drives_report_dimensions_and_parameter_labels(self):
        """替换模型规格后，报告尺寸、参数量与内嵌数据应随实际配置变化。"""
        vocab = {**TOY_VOCAB, "extra": 6}
        config = {**TOY_CONFIG, "vocab_size": 7, "context_length": 6, "emb_dim": 6, "n_heads": 3}
        with tempfile.TemporaryDirectory() as output_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = main(output_dir, config=config, vocab=vocab)

            with open(os.path.join(output_dir, "gpt2toy_dissection.md"), encoding="utf-8") as stream:
                markdown = stream.read()
            with open(os.path.join(output_dir, "gpt2toy_inspector.html"), encoding="utf-8") as stream:
                html = stream.read()

        self.assertEqual(payload["config"]["context_length"], 6)
        self.assertEqual(payload["config"]["emb_dim"], 6)
        self.assertEqual(payload["config"]["n_heads"], 3)
        self.assertEqual(payload["config"]["head_dim"], 2)
        self.assertEqual(payload["config"]["ffn_hidden_dim"], 24)
        self.assertEqual(payload["param_stats"]["total_p"], 624)
        self.assertEqual(len(payload["weights"]["pos_emb_weight"]), 6)
        self.assertEqual(len(payload["weights"]["pos_emb_weight"][0]), 6)
        self.assertEqual(len(payload["weights"]["blocks"][0]["causal_mask"]), 6)
        self.assertEqual(len(payload["weights"]["tok_emb_weight"]), 7)
        self.assertIn("Dim 5", markdown)
        self.assertIn("Head 2", markdown)
        self.assertIn("624", markdown)
        self.assertIn("624", html)
        self.assertIn("24", markdown)
        self.assertNotIn("**312**", markdown)
        html_payload = embedded_trace(html)
        self.assertEqual(html_payload["config"]["context_length"], 6)
        self.assertEqual(html_payload["config"]["emb_dim"], 6)
        self.assertEqual(html_payload["config"]["n_heads"], 3)
        self.assertEqual(html_payload["param_stats"]["total_p"], 624)
        self.assertEqual(len(html_payload["weights"]["pos_emb_weight"]), 6)
        self.assertEqual(len(html_payload["weights"]["pos_emb_weight"][0]), 6)

    def test_invalid_repeat_positions_and_nonmatching_tokens_raise(self):
        """同词异位探针拒绝越界、负索引和两个 Token ID 不同的位置。"""
        model, tokenizer = make_default_model_and_tokenizer()
        with self.assertRaises(ValueError):
            collect_same_token_example(model, tokenizer, "Hello I", positions=(2, 4))
        with self.assertRaises(ValueError):
            collect_same_token_example(model, tokenizer, REPEATED_CONTEXT, positions=(2, 3))
        with self.assertRaises(ValueError):
            collect_same_token_example(model, tokenizer, REPEATED_CONTEXT, positions=(-1, 4))


if __name__ == "__main__":
    unittest.main()
