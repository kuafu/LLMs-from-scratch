import contextlib
import copy
import io
import unittest
from unittest import mock

import torch

from gpt2toy import (
    FeedForward,
    GELU,
    GPTModel,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
    generate_text_simple,
)
from gpt2toy_data import GPTDatasetV1, SimpleTokenizerV2
from gpt2toy_diagnostics import TeachingTrace


def make_model(*, n_layers=2, drop_rate=0.0, context_length=8, n_heads=2, emb_dim=8):
    """构建固定随机种子的微型模型，供形状、因果性及梯度对照测试使用。"""
    torch.manual_seed(41)
    return GPTModel({
        "vocab_size": 12,
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "drop_rate": drop_rate,
        "qkv_bias": False,
    })


def contains_tensor(value):
    """递归检查快照是否残留 Tensor，防止长期保留计算图或可变张量引用。"""
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(contains_tensor(k) or contains_tensor(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(contains_tensor(item) for item in value)
    return False


class GPTToyRegressionTests(unittest.TestCase):
    """验证 GPT 张量路径、因果约束、自回归生成及可选 trace 的非干扰性。"""

    def _assert_forward_boundaries(self, module, inputs, trace):
        """用 PyTorch 实际调用的前后钩子核对日志顺序、嵌套深度和真实张量形状。"""
        watched = (GPTModel, TransformerBlock, MultiHeadAttention, LayerNorm, FeedForward, GELU)
        calls, active, handles = [], [], []

        def entering(current, args):
            calls.append(("begin", type(current).__name__, len(active), tuple(args[0].shape)))
            active.append(current)

        def leaving(current, args, output):
            self.assertIs(active.pop(), current)
            calls.append(("end", type(current).__name__, len(active), tuple(output.shape)))

        with torch.no_grad():
            expected = module(inputs)
        for child in module.modules():
            if isinstance(child, watched):
                handles.append(child.register_forward_pre_hook(entering))
                handles.append(child.register_forward_hook(leaving))
        stdout = io.StringIO()
        try:
            with torch.no_grad(), contextlib.redirect_stdout(stdout):
                actual = module(inputs, trace=trace)
        finally:
            for handle in handles:
                handle.remove()

        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(active, [])
        expected_markers = []
        for phase, name, depth, shape in calls:
            arrow = ">>>" if phase == "begin" else "<<<"
            label = ("in_idx" if phase == "begin" else "logits") if name == "GPTModel" else ("x" if phase == "begin" else "out")
            indent = "  " + "    " * depth
            expected_markers.append(f"{indent}{arrow} {name}.forward {phase:<5} | {label}.shape={shape}")
        actual_markers = [line for line in stdout.getvalue().splitlines()
                          if line.lstrip().startswith((">>> ", "<<< "))]
        self.assertEqual(actual_markers, expected_markers)

    def test_forward_markers_follow_actual_nested_calls_and_trace_reuse(self):
        """多层和零层模型都按实际父调用缩进，最终 LN 与 Block 保持同级。"""
        for n_layers in (0, 2):
            with self.subTest(n_layers=n_layers):
                model = make_model(n_layers=n_layers).eval()
                trace = TeachingTrace(verbose=True, show_data=False)
                for inputs in (torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])):
                    self._assert_forward_boundaries(model, inputs, trace)

    def test_standalone_forward_markers_start_at_root_when_trace_is_reused(self):
        """子模块及 GELU 可独立调用，顺序复用观测器时重新建立根层级和输入标签。"""
        block = make_model(n_layers=1).eval().trf_blocks[0]
        trace = TeachingTrace(verbose=True, show_data=True)
        for module, length in ((block, 2), (block, 5), (block.att, 3), (block.norm1, 4), (block.ff, 2), (block.ff.layers[1], 3)):
            with self.subTest(module=type(module).__name__, length=length):
                inputs = torch.arange(2 * length * 8, dtype=torch.float32).view(2, length, 8) / 10
                self._assert_forward_boundaries(module, inputs, trace)

    def test_untraced_forward_and_batched_generation_are_silent_and_stay_on_tensor_path(self):
        """不启用 trace 时，推理应保持静默且不做 CPU 复制或列表转换。"""
        model = make_model().eval()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected cpu copy")), \
                    mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("unexpected list conversion")):
                logits = model(ids)
                generated = generate_text_simple(model, ids, max_new_tokens=2, context_size=4)

        self.assertEqual(tuple(logits.shape), (2, 3, 12))
        self.assertEqual(tuple(generated.shape), (2, 5))
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_training_trace_preserves_dropout_outputs_gradients_and_rng(self):
        """启用观测不应改变 Dropout 的随机数序列、前向结果或反向传播梯度。"""
        plain = make_model(drop_rate=0.25).train()
        traced = copy.deepcopy(plain).train()
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

        torch.manual_seed(987)
        plain_out = plain(ids)
        plain_out.square().mean().backward()
        plain_rng = torch.random.get_rng_state().clone()

        trace = TeachingTrace(verbose=False, show_data=False)
        torch.manual_seed(987)
        traced_out = traced(ids, trace=trace)
        traced_out.square().mean().backward()
        traced_rng = torch.random.get_rng_state().clone()

        self.assertTrue(torch.equal(plain_out, traced_out))
        self.assertTrue(torch.equal(plain_rng, traced_rng))
        plain_grads = dict(plain.named_parameters())
        traced_grads = dict(traced.named_parameters())
        self.assertEqual(plain_grads.keys(), traced_grads.keys())
        for name in plain_grads:
            self.assertIsNotNone(plain_grads[name].grad, name)
            self.assertTrue(torch.equal(plain_grads[name].grad, traced_grads[name].grad), name)
        self.assertFalse(contains_tensor(trace.record))

    def test_reused_trace_resets_and_keeps_old_python_snapshots(self):
        """顺序复用观测器时清空当前记录，同时保持旧快照的数值独立性。"""
        first_ids = torch.tensor([[1, 2, 3]])
        second_ids = torch.tensor([[7, 8, 9]])
        for n_layers in (0, 2):
            with self.subTest(n_layers=n_layers):
                model = make_model(n_layers=n_layers).eval()
                trace = TeachingTrace(verbose=False, show_data=False)
                model(first_ids, trace=trace)
                old_record = trace.record
                old_snapshot = copy.deepcopy(old_record)
                self.assertEqual(len(old_record["blocks"]), n_layers)
                if n_layers:
                    block = old_record["blocks"][0]
                    self.assertTrue({"norm1", "att", "res1", "norm2", "ffn", "res2"}.issubset(block))

                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.125)
                model(second_ids, trace=trace)

                self.assertIsNot(trace.record, old_record)
                self.assertEqual(old_record, old_snapshot)
                self.assertEqual(trace.record["embeddings"]["in_idx"], [7, 8, 9])
                self.assertEqual(len(trace.record["blocks"]), n_layers)
                self.assertFalse(contains_tensor(trace.record))

    def test_causal_prefix_outputs_do_not_depend_on_later_tokens(self):
        """相同位置的历史前缀不得读取未来 Token，验证因果自注意力约束。"""
        model = make_model(n_layers=2, context_length=6).eval()
        full = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            full_logits = model(full)
            prefix_logits = model(full[:, :3])
        torch.testing.assert_close(full_logits[:, :3], prefix_logits, rtol=1e-6, atol=1e-7)

        trace = TeachingTrace(verbose=False, show_data=False)
        model(full, trace=trace)
        self.assertEqual(trace.record["embeddings"]["pos_indices"], [0, 1, 2, 3, 4])

    def test_verbose_trace_displays_all_configured_attention_heads(self):
        """单头和多头配置都应完整显示实际注意力头，不能固定读取第二个头。"""
        ids = torch.tensor([[1, 2, 3]])
        for n_heads in (1, 3):
            with self.subTest(n_heads=n_heads):
                model = make_model(n_layers=1, n_heads=n_heads, emb_dim=6).eval()
                trace = TeachingTrace(verbose=True, show_data=True)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    logits = model(ids, trace=trace)

                self.assertEqual(tuple(logits.shape), (1, 3, 12))
                self.assertEqual(len(trace.record["blocks"][0]["att"]["heads"]), n_heads)
                display = stdout.getvalue()
                for head_idx in range(n_heads):
                    self.assertIn(f"Head{head_idx}=", display)
                if n_heads == 1:
                    self.assertNotIn("Head1=", display)

    def test_batched_generation_crops_context_and_zero_tokens_preserve_prompt(self):
        """验证整个 batch 的贪婪生成、滑动窗口裁剪及零步生成边界。"""
        model = make_model().eval()
        prompt = torch.tensor([[1, 2], [7, 8]])
        observed_contexts = []

        def capture_context(event, **values):
            if event == "generation_start":
                observed_contexts.append(values["idx_cond"].clone())

        actual = generate_text_simple(
            model, prompt, max_new_tokens=4, context_size=3, trace=capture_context,
        )
        expected = prompt.clone()
        expected_contexts = []
        with torch.no_grad():
            for _ in range(4):
                context = expected[:, -3:]
                expected_contexts.append(context.clone())
                logits = model(context)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                expected = torch.cat((expected, next_token), dim=1)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual([ctx.shape[1] for ctx in observed_contexts], [2, 3, 3, 3])
        for observed, reference in zip(observed_contexts, expected_contexts):
            self.assertTrue(torch.equal(observed, reference))

        no_step_events = []
        unchanged = generate_text_simple(
            model, prompt, max_new_tokens=0, context_size=3,
            trace=lambda event, **values: no_step_events.append(event),
        )
        self.assertTrue(torch.equal(unchanged, prompt))
        self.assertEqual(no_step_events, [])

    def test_tokenizer_unknown_punctuation_and_sliding_window_shift(self):
        """验证未知词/标点映射，以及 CLM（Causal Language Modeling）目标右移一位。"""
        vocab = {
            "<|unk|>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
            ",": 7, "!": 8,
        }
        tokenizer = SimpleTokenizerV2(vocab)
        encoded = tokenizer.encode("a, missing!")
        self.assertEqual(encoded, [1, 7, 0, 8])
        self.assertEqual(tokenizer.decode(encoded), "a, <|unk|>!")

        dataset = GPTDatasetV1("a b c d e f", tokenizer, max_length=3, stride=2)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(torch.equal(dataset[0][0], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(dataset[0][1], torch.tensor([2, 3, 4])))
        self.assertTrue(torch.equal(dataset[1][0], torch.tensor([3, 4, 5])))
        self.assertTrue(torch.equal(dataset[1][1], torch.tensor([4, 5, 6])))


if __name__ == "__main__":
    unittest.main()
