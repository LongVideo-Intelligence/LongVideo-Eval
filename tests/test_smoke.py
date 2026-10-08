"""End-to-end smoke test: the single-shot pipeline runs and meters cost, no hardware.

Proves the architecture (decode -> select -> encode -> prune -> generate) is wired and the
cost accounting accumulates correctly. Real components (decode/encode/backend) are skeletons
until ported; these use the test doubles in longvideo_eval.testing.
"""
from longvideo_eval.interfaces import Budget, Sample
from longvideo_eval.testing import FakeBackend, FakeDecoder, FakeEncoder
from longvideo_eval.frontend.select.uniform import UniformSelector
from longvideo_eval.frontend.prune.identity import IdentityPruner
from longvideo_eval.orchestration.single_shot import SingleShotOrchestrator
from longvideo_eval.runners.evaluate import accuracy, evaluate


def _orch():
    return SingleShotOrchestrator(
        FakeDecoder(n=64), UniformSelector(), FakeEncoder(), IdentityPruner(), FakeBackend()
    )


def test_single_shot_runs_and_meters():
    budget = Budget(frame_count=8, resolution=0.5, decode_budget=64)  # r = scale factor (per side)
    ans = _orch().run(Sample("v1", "/x.mp4", "what happens?", subtitles="hi"), budget)

    assert ans.text == "A"
    c = ans.cost
    assert c.frames_decoded == 64            # decode charged the whole pool
    assert c.frames_encoded == 8             # only selected frames encoded
    assert c.prefill_tokens == 8 * 64        # 64 tokens/frame kept (identity pruner)
    assert c.rounds == 0                     # single shot
    assert c.decode_seconds > 0 and c.wall_seconds >= 0


def test_uniform_selector_respects_frame_count():
    budget = Budget(frame_count=8, resolution=0.25, decode_budget=64)  # r = scale factor (per side)
    ans = _orch().run(Sample("v2", "/y.mp4", "q"), budget)
    assert ans.cost.frames_encoded == 8


def test_evaluate_scores_mcq():
    budget = Budget(frame_count=4, decode_budget=16)
    samples = [Sample("v1", "/a.mp4", "q", choices=["A", "B", "C", "D"])]
    results = evaluate(_orch(), samples, budget, golds={"v1": "A"})
    assert results[0].correct is True
    assert accuracy(results) == 1.0


def test_mcq_scoring_handles_verbose_answers():
    from longvideo_eval.runners.evaluate import extract_choice, score_mcq
    assert extract_choice("The answer is A") == "A"
    assert extract_choice("(B)") == "B"
    assert extract_choice("C. because the person leaves") == "C"
    assert score_mcq("The answer is A.", "A") is True
    assert score_mcq("B", "A") is False
    # A [[X]]/{{X}} wrap takes PRIORITY over positional heuristics — the anchor for verbose /
    # thinking models. The LAST wrap wins (final choice after reasoning that mentions others).
    assert extract_choice("Considering (A) and (C), the answer is [[C]]") == "C"
    assert extract_choice("Option A looks plausible ... final: {{D}}") == "D"
    assert extract_choice("First I thought [[A]] but actually [[B]]") == "B"  # last wrap
    assert extract_choice("[[b]]") == "B"                                     # case-normalized
    # No wrap -> the positional fallback.
    assert extract_choice("The answer is A") == "A"


def test_frame_count_zero_selects_none():
    budget = Budget(frame_count=0, decode_budget=16)
    ans = _orch().run(Sample("v3", "/z.mp4", "q"), budget)
    assert ans.cost.frames_encoded == 0        # 0 means 0, not the whole pool
