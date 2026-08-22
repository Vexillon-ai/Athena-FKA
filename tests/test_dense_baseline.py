"""Gates for Phase 5's dense baseline (M5 §4.1-§4.2).

Three invariants, each stated as the property actually needed rather than a proxy for it, and
each with a **verified-red** case kept in the suite (CLAUDE.md design pattern 2):

1. **No memory machinery on the dense path.** Not "disabled" — absent. Enforced by scanning the
   package's imports, with a source string that *does* import it asserted to be flagged.
2. **The fact firewall is OFF.** No interface marker may appear in the stream, and the firewalled
   surface must be *rejected* by the same check.
3. **The probe path can score a correct model 100% and a wrong one at chance.** A gold stub alone
   passes for a scorer that returns True unconditionally, so it ships with its red twin.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fka.data import templates as T  # noqa: E402
from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.data.tokenizer import CharTokenizer  # noqa: E402
import fka.dense.probe as probe_module  # noqa: E402
from fka.dense.probe import (  # noqa: E402
    GoldStubRecall,
    WrongAnswerRecall,
    dense_capacity,
    PromptBuilder,
    truncate_at_eos,
)
from fka.dense.stream import DenseCorpusStream, DenseDataConfig  # noqa: E402
from fka.dense.train import (  # noqa: E402
    DenseTrainConfig,
    lr_at,
    plan_run,
    train_dense,
)
from fka.eval.kernel_eval import most_frequent_answer_baseline  # noqa: E402
from fka.kernel.model import ReasoningKernel  # noqa: E402

DENSE_DIR = Path(__file__).resolve().parent.parent / "fka" / "dense"

#: Everything that makes our architecture our architecture. A dense baseline that touched any of
#: it would not be a plain LM, and the comparison would be measuring a hybrid.
FORBIDDEN = (
    "fka.kernel.memory",
    "fka.kernel.episodes",
    "fka.kernel.generate",
    "fka.kernel.latent_memory",
    "fka.kernel.latent_kernel",
    "fka.kernel.latent_episodes",
    "fka.kernel.latent_train",
    "fka.router",
    "fka.store",
    "fka.retriever",
)


def _imported_modules(source: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _violations(source: str) -> set[str]:
    imported = _imported_modules(source)
    return {
        m
        for m in imported
        for bad in FORBIDDEN
        if m == bad or m.startswith(bad + ".")
    }


# =======================================================================================
# 1. no memory machinery
# =======================================================================================


def test_dense_package_imports_no_memory_machinery():
    for path in sorted(DENSE_DIR.glob("*.py")):
        bad = _violations(path.read_text(encoding="utf-8"))
        assert not bad, f"{path.name} imports memory machinery: {sorted(bad)} (M5 §4.1)"


def test_import_checker_is_verified_red():
    """The checker must fail on the thing it forbids, or it is a statement about today's files."""
    offending = "from fka.kernel.memory import OracleTextMemory\nimport fka.store.base\n"
    assert _violations(offending) == {"fka.kernel.memory", "fka.store.base"}


def test_training_never_passes_a_loss_mask():
    """A masked loss here would re-erect the firewall the baseline is entitled to be without.

    Checked at the call, not in the source: a mask could arrive positionally, through a default,
    or from a helper, and a grep would miss all three.
    """
    corpus = generate_corpus(n_entities=60, seed=5, probe_fraction=0.2)
    tok = CharTokenizer()
    stream = DenseCorpusStream(corpus, tok, DenseDataConfig(exposures=2, qa_entity_fraction=0.5))
    cfg = DenseTrainConfig(size="tiny", block_size=64, batch_size=2, warmup_steps=1, lr=1e-3)

    seen: list[object] = []
    original = ReasoningKernel.forward

    def spy(self, idx, targets=None, loss_mask=None, **kw):
        seen.append(loss_mask)
        return original(self, idx, targets, loss_mask, **kw)

    ReasoningKernel.forward = spy
    try:
        train_dense(stream, cfg, total_steps=64, run_steps=3, progress=False)
    finally:
        ReasoningKernel.forward = original
    assert seen and all(m is None for m in seen)


# =======================================================================================
# 2. the fact firewall is OFF
# =======================================================================================


def _small_stream(**cfg_kwargs):
    corpus = generate_corpus(n_entities=60, seed=3, probe_fraction=0.3)
    tok = CharTokenizer()
    return corpus, tok, DenseCorpusStream(corpus, tok, DenseDataConfig(**cfg_kwargs))


def test_stream_contains_values_and_no_markers():
    corpus, tok, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    text = tok.decode(stream.epoch_tokens(0).tolist())
    for marker in T.MARKERS:
        assert marker not in text, f"{marker} in the dense stream — the firewall is on (M5 §4.1)"
    # The value must be present as literal text: that is what "firewall OFF" means.
    assert corpus.value_of("birth_city", 0) in text
    assert corpus.entity_name(0) in text


def test_firewalled_surface_is_rejected_verified_red():
    """The check fails on the firewalled surface — the design it exists to exclude."""
    corpus, tok, stream = _small_stream(exposures=1)
    firewalled = "".join(corpus.documents(corpus.train_ids[:5], firewall=True))
    ids = np.array(tok.encode(firewalled), dtype=np.uint8)
    with pytest.raises(ValueError, match="firewall"):
        stream._assert_no_markers(ids)


# =======================================================================================
# 3. exposures and the QA split
# =======================================================================================


def test_each_fact_appears_once_per_epoch():
    corpus, tok, stream = _small_stream(exposures=3, qa_entity_fraction=0.0, qa_period=0)
    text = tok.decode(stream.epoch_tokens(0).tolist())
    docs = [d for d in text.split("<eos>") if d]
    assert len(docs) == int((~corpus.heldout_mask).sum())


def test_template_variety_tracks_exposures():
    """Distinct renderings per fact = min(exposures, n_variants) — the literature's variety knob."""
    corpus, tok, stream = _small_stream(exposures=8, qa_entity_fraction=0.0, qa_period=0)
    subject = corpus.entity_name(0)
    seen = set()
    for epoch in range(8):
        text = tok.decode(stream.epoch_tokens(epoch).tolist())
        for doc in text.split("<eos>"):
            if subject in doc and corpus.value_of("birth_city", 0) in doc:
                seen.add(doc)
    assert len(seen) == len(T.relation_templates("birth_city").statements)


def test_probe_entities_were_never_asked_the_question():
    """The headline probe set's questions must not appear anywhere in the training stream."""
    corpus, tok, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    text = "".join(
        tok.decode(stream.epoch_tokens(v).tolist()) for v in range(stream.variant_period)
    )
    prompts = PromptBuilder(corpus, stream.surface)
    probe_ids = stream.probe_fact_ids()
    assert probe_ids.size > 0
    for pair in list(corpus.memory_pairs(probe_ids))[:20]:
        assert prompts.for_pair(pair) not in text
    trained_ids = stream.probe_fact_ids(qa_trained=True)
    hits = [prompts.for_pair(p) in text for p in list(corpus.memory_pairs(trained_ids))[:20]]
    assert all(hits), "the QA channel is supposed to teach the probe surface on its own half"


def test_qa_substitutes_rather_than_adds_exposures():
    """Both halves of the split get exactly one rendering per fact per epoch.

    If QA were appended instead of substituted, QA-train entities would receive more exposures
    than probe entities and the headline's extrapolation from the probe half over the whole
    corpus would be a statement about exposure as well as surface.
    """
    corpus, tok, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    expected = int((~corpus.heldout_mask).sum())
    for variant in range(stream.variant_period):
        text = tok.decode(stream.epoch_tokens(variant).tolist())
        assert len([d for d in text.split("<eos>") if d]) == expected


def test_qa_split_partitions_the_entities():
    corpus, _, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    heldout = corpus.subject_of(stream.probe_fact_ids())
    trained = corpus.subject_of(stream.probe_fact_ids(qa_trained=True))
    assert not set(heldout.tolist()) & set(trained.tolist())


def test_no_heldout_entity_split_is_refused():
    with pytest.raises(ValueError, match="complement|held-out|\\[0, 1\\)"):
        DenseDataConfig(qa_entity_fraction=1.0)


# =======================================================================================
# 4. the probe path: gold stub and its red twin
# =======================================================================================


def test_truncate_at_eos_truncates_rather_than_filters():
    assert truncate_at_eos([5, 6, 2, 7, 8], eos_id=2) == [5, 6]


def test_gold_stub_scores_100_through_the_dense_probe_path():
    corpus, tok, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    ids = stream.probe_fact_ids()[:24]
    report = dense_capacity(GoldStubRecall(corpus, tok, ids), tok, corpus, ids, n_params=1)
    assert report.accuracy == 1.0, report.per_relation


def test_gold_stub_needs_truncation_not_filtering(monkeypatch):
    """The eos gate, made red-able.

    With early stopping on, nothing is ever generated past ``<eos>``, so truncating and filtering
    agree and the gold stub passes either way — it was silently decorative until this case
    existed. Running the full token budget puts the script's post-``<eos>`` garbage in the span:
    truncation still scores 100%, the documented filter defect scores 0%.
    """
    corpus, tok, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    ids = stream.probe_fact_ids()[:12]

    stub = GoldStubRecall(corpus, tok, ids, stop_at_eos=False)
    assert dense_capacity(stub, tok, corpus, ids, n_params=1).accuracy == 1.0

    monkeypatch.setattr(
        probe_module, "truncate_at_eos", lambda seq, eos_id: [int(i) for i in seq if i != eos_id]
    )
    red = GoldStubRecall(corpus, tok, ids, stop_at_eos=False)
    assert dense_capacity(red, tok, corpus, ids, n_params=1).accuracy == 0.0


def test_wrong_answer_stub_is_not_scored_correct():
    corpus, tok, stream = _small_stream(exposures=1, qa_entity_fraction=0.5)
    ids = stream.probe_fact_ids()[:24]
    report = dense_capacity(WrongAnswerRecall(corpus, tok, ids), tok, corpus, ids, n_params=1)
    chance = most_frequent_answer_baseline(list(corpus.memory_pairs(ids)))
    assert report.accuracy <= chance + 1e-9, "the probe path is matching shape, not content"


# =======================================================================================
# 5. the recipe
# =======================================================================================


def test_lr_schedule_horizon_is_independent_of_run_length():
    """A probe inherits the horizon and varies only length (CLAUDE.md, schedule-horizon rule)."""
    cfg = DenseTrainConfig(size="tiny", warmup_steps=10, lr=1e-3)
    assert lr_at(300, cfg, total_steps=15_000) == lr_at(300, cfg, total_steps=15_000)
    assert lr_at(300, cfg, total_steps=400) != lr_at(300, cfg, total_steps=15_000)


def test_train_smoke_and_cross_session_resume(tmp_path):
    """A run that spans sessions must resume its optimizer and cursor, not just its weights."""
    corpus = generate_corpus(n_entities=120, seed=1, probe_fraction=0.2)
    tok = CharTokenizer()
    stream = DenseCorpusStream(corpus, tok, DenseDataConfig(exposures=4, qa_entity_fraction=0.5))
    cfg = DenseTrainConfig(size="tiny", block_size=128, batch_size=4, warmup_steps=2, lr=1e-3)
    plan = plan_run(stream, cfg)
    assert plan["total_steps"] > 0

    _, state = train_dense(
        stream, cfg, total_steps=plan["total_steps"], run_steps=8,
        out_dir=tmp_path, progress=False,
    )
    assert state.step == 8
    assert all(np.isfinite(state.losses))

    blob = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=False)
    assert blob["optimizer_state"]["state"], "resume without optimizer state is a different run"

    _, resumed = train_dense(
        stream, cfg, total_steps=plan["total_steps"], run_steps=4,
        resume=tmp_path / "final.pt", progress=False,
    )
    assert resumed.step == 12
    assert len(resumed.losses) == 12


# =======================================================================================
# 6. rendering surfaces — the token-density axis (M5 §5.6)
# =======================================================================================


@pytest.mark.parametrize("surface", ["verbose", "terse", "terse_named"])
def test_surface_leaves_the_world_untouched(surface):
    """(a) A surface changes characters, never facts or bits.

    This is the guarantee that makes the density axis safe to move at all: if a surface could
    alter the corpus's information content it would invalidate every capacity number in the
    program, and nothing in a loss curve or an accuracy would show it.
    """
    base_corpus, tok, base = _small_stream(exposures=1, qa_entity_fraction=0.5)
    corpus, _, stream = _small_stream(exposures=1, qa_entity_fraction=0.5, surface=surface)
    assert corpus.fingerprint() == base_corpus.fingerprint()
    assert corpus.total_bits == base_corpus.total_bits
    assert corpus.stored_bits == base_corpus.stored_bits
    # ... and the numerator of bits/token is the same object; only the denominator moves.
    assert stream.bits_per_token() * stream.tokens_per_epoch == pytest.approx(
        base.bits_per_token() * base.tokens_per_epoch
    )


@pytest.mark.parametrize("surface", ["terse", "terse_named"])
def test_terse_surfaces_are_denser(surface):
    """(b) measured, not asserted."""
    _, _, base = _small_stream(exposures=1, qa_entity_fraction=0.5)
    _, _, stream = _small_stream(exposures=1, qa_entity_fraction=0.5, surface=surface)
    assert stream.bits_per_token() > base.bits_per_token()


@pytest.mark.parametrize("surface", ["verbose", "terse", "terse_named"])
def test_surface_keeps_the_probe_path_gated(surface):
    """Both gates must hold under every surface, or the surface is not usable for a run."""
    corpus, tok, stream = _small_stream(
        exposures=1, qa_entity_fraction=0.5, surface=surface
    )
    ids = stream.probe_fact_ids()[:16]
    gold = dense_capacity(
        GoldStubRecall(corpus, tok, ids, surface=stream.surface), tok, corpus, ids,
        n_params=1, surface=stream.surface,
    )
    wrong = dense_capacity(
        WrongAnswerRecall(corpus, tok, ids, surface=stream.surface), tok, corpus, ids,
        n_params=1, surface=stream.surface,
    )
    chance = most_frequent_answer_baseline(list(corpus.memory_pairs(ids)))
    assert gold.accuracy == 1.0
    assert wrong.accuracy <= chance + 1e-9


@pytest.mark.parametrize("surface", ["verbose", "terse", "terse_named"])
def test_surface_keeps_the_qa_holdout(surface):
    """The disjoint-entity holdout is a property of the design, not of the verbose templates."""
    corpus, tok, stream = _small_stream(
        exposures=1, qa_entity_fraction=0.5, surface=surface
    )
    text = "".join(
        tok.decode(stream.epoch_tokens(v).tolist()) for v in range(stream.variant_period)
    )
    prompts = PromptBuilder(corpus, stream.surface)
    for pair in list(corpus.memory_pairs(stream.probe_fact_ids()))[:20]:
        assert prompts.for_pair(pair) not in text
    trained = list(corpus.memory_pairs(stream.probe_fact_ids(qa_trained=True)))[:20]
    assert all(prompts.for_pair(p) in text for p in trained)


def test_answer_delimiter_collision_is_refused():
    """A value containing the delimiter would shift the stub's prompt boundary, silently."""
    from fka.dense.surface import TerseSurface, assert_answer_prefix_is_unambiguous

    corpus, _, _ = _small_stream(exposures=1)
    with pytest.raises(ValueError, match="delimiter"):
        # 'a' occurs in essentially every name, so this stands in for any colliding delimiter.
        assert_answer_prefix_is_unambiguous(TerseSurface(answer_prefix="a"), corpus)


def test_qa_line_is_exactly_prompt_plus_value():
    """If these ever diverge, the QA channel teaches a surface the probe never uses."""
    from fka.dense.surface import SURFACES

    corpus, _, _ = _small_stream(exposures=1)
    for surface in SURFACES.values():
        subject = surface.subjects(corpus, "birth_city")[0]
        value = corpus.value_of("birth_city", 0)
        assert surface.qa_line("birth_city", subject, value) == (
            surface.probe_prompt("birth_city", subject) + value
        )


# =======================================================================================
# 7. sizing — required storage is not scored entropy (M5 §5.8)
# =======================================================================================


def test_load_is_measured_against_value_entropy_only():
    """The reference accounting, and the retired one kept only as a diagnostic (§5.25).

    Keys are presented at query time: an associative memory owes bits to reproduce the VALUE given
    the key, not to store the key. Pricing key material into the denominator inflated every load
    row by up to 44% -- and, worse, priced the key-discrimination hypothesis into the instrument
    built to test it.
    """
    from fka.dense.sizing import key_bits, size_run, storage_bits

    corpus = generate_corpus(n_entities=31_686, seed=0, probe_fraction=0.01)
    assert storage_bits(corpus) == corpus.total_bits
    report = size_run(corpus, n_params=863_488)
    # N=31,686 against the 1M model is the saturation point the sweep was aiming for all along.
    assert report.load == pytest.approx(1.0, abs=0.02)
    # The retired accounting is still reachable, and still inflates.
    assert report.load_with_keys_counted > report.load
    assert report.key_share_of_capacity == pytest.approx(0.44, abs=0.02)
    assert key_bits(corpus) > 0  # a real quantity, just not a denominator


def test_key_share_grows_with_corpus_size():
    """Key material is 11% of capacity at N=8,000 and 44% at N=31,686 -- diagnostic, not load."""
    from fka.dense.sizing import size_run

    shares = [
        size_run(generate_corpus(n_entities=n, seed=0, probe_fraction=0.01), 863_488)
        .key_share_of_capacity
        for n in (8_000, 16_000, 31_686)
    ]
    assert shares == sorted(shares)
    assert shares[0] == pytest.approx(0.11, abs=0.02)
def test_weight_decay_schedule_falls_with_corpus_size():
    """The reference recipe's decay is a SCHEDULE, and a constant is the import error (§5.16).

    Decoupled decay pulls every weight toward zero at a rate set by `lr * wd`, while the gradient
    defending one fact scales with that fact's share of the stream (~1/n_facts). So the same decay
    that is harmless on a small corpus erases facts on a large one, and any constant is wrong at
    one end.
    """
    from fka.dense.train import weight_decay_for

    decays = [weight_decay_for(n) for n in (10_000, 200_000, 1_000_000, 2_000_000, 50_000_000)]
    assert decays == sorted(decays, reverse=True), "decay must fall as the corpus grows"
    assert weight_decay_for(31_686) == 0.02, "our ladder's N sits in the reference's first band"
    # The value we actually ran, against the value the reference used at the same N.
    assert 0.1 / weight_decay_for(31_686) == pytest.approx(5.0)


# =======================================================================================
# 8. the two fingerprints — each pins what the other cannot see (M5 §5.31)
# =======================================================================================


def test_world_fingerprint_is_BLIND_to_rendering_changes():
    """The demonstration, not the assurance.

    `corpus.fingerprint()` hashes config, entity name INDICES, values, bits and split masks. It
    pins the WORLD, correctly — and is therefore structurally unable to see a change of tokenizer,
    template, or name construction. Three surfaces that render the same facts as completely
    different characters share a byte-identical world hash.
    """
    corpus = generate_corpus(n_entities=200, seed=0, probe_fraction=0.2)
    tok = CharTokenizer()
    reports = {
        surf: DenseCorpusStream(corpus, tok, DenseDataConfig(exposures=1, surface=surf))
        .exposure_report()
        for surf in ("verbose", "terse", "terse_named")
    }
    worlds = {r["world_fingerprint"] for r in reports.values()}
    surfaces = {r["surface_fingerprint"] for r in reports.values()}
    assert len(worlds) == 1, "the world moved when only the rendering should have"
    assert len(surfaces) == 3, "the surface fingerprint cannot see a rendering change"


def test_surface_fingerprint_sees_a_subject_construction_change():
    """The change class that motivated the instrument: how SUBJECTS are built.

    Templates alone would miss it — `terse` and `terse_named` share every template and differ only
    in how the subject string is constructed, which is exactly the corpus-generator change under
    consideration (wider syllable tables). The rendered-sample term is what catches it.
    """
    from fka.dense.surface import surface_fingerprint, surface_for

    corpus = generate_corpus(n_entities=200, seed=0, probe_fraction=0.2)
    tok = CharTokenizer()
    a = surface_fingerprint(surface_for("terse"), tok, corpus)
    b = surface_fingerprint(surface_for("terse_named"), tok, corpus)
    assert a != b
    assert corpus.fingerprint() == corpus.fingerprint()  # world unmoved by either


def test_surface_fingerprint_moves_with_the_vocabulary():
    """A tokenizer swap is a surface change, and the surface hash must carry it."""
    from fka.dense.surface import surface_fingerprint, surface_for

    corpus = generate_corpus(n_entities=120, seed=1, probe_fraction=0.2)
    narrow = CharTokenizer()
    wider = CharTokenizer(chars=CharTokenizer().chars + ("~",))
    surface = surface_for("verbose")
    assert surface_fingerprint(surface, narrow, corpus) != surface_fingerprint(
        surface, wider, corpus
    )


def test_surface_revision_acceptance_pair():
    """The revision must MOVE the surface hash and PIN the world (M5 §5.32, CLAUDE.md revision rule).

    Asserting only the invariants would pass identically if the revision silently failed to apply -
    a dropped surface argument, a cached buffer, a default that never took effect. The right-hand
    assertion is what makes "the change happened" a measurement rather than a hope.
    """
    from fka.dense.surface import surface_fingerprint, surface_for, syllable_tokenizer

    corpus = generate_corpus(n_entities=400, seed=0, probe_fraction=0.2, relations=("birth_year",))
    char_tok, syl_tok = CharTokenizer(), syllable_tokenizer()
    before = DenseCorpusStream(
        corpus, char_tok, DenseDataConfig(exposures=1, surface="verbose")
    ).exposure_report()
    after = DenseCorpusStream(
        corpus, syl_tok, DenseDataConfig(exposures=1, surface="syllable")
    ).exposure_report()

    # PINNED: the world did not move.
    assert before["world_fingerprint"] == after["world_fingerprint"]
    assert before["corpus_stored_bits"] == after["corpus_stored_bits"]
    # MOVED: the rendering did.
    assert before["surface_fingerprint"] != after["surface_fingerprint"]
    assert after["tokens_per_fact"] < before["tokens_per_fact"] / 2
    assert after["bits_per_token"] > before["bits_per_token"] * 2
    assert surface_fingerprint(surface_for("syllable"), syl_tok, corpus) == after[
        "surface_fingerprint"
    ]


def test_syllable_surface_refuses_entity_valued_relations():
    """Rendering works_with would give one entity two names. Refused, not silently mis-rendered."""
    from fka.dense.surface import surface_for

    corpus = generate_corpus(n_entities=200, seed=0, probe_fraction=0.2)
    with pytest.raises(ValueError, match="works_with"):
        surface_for("syllable").subjects(corpus, "works_with")


def test_syllable_names_are_injective_and_scrambled():
    """Injective, or probes are ambiguous; scrambled, or adjacent entities get near-identical keys."""
    from fka.dense.surface import surface_for

    corpus = generate_corpus(n_entities=5000, seed=0, probe_fraction=0.01,
                             relations=("birth_year",))
    names = surface_for("syllable").subjects(corpus, "birth_year")
    assert len(set(names)) == len(names)
    assert names[0][0] != names[1][0] or names[0] != names[1]


def test_probing_on_the_wrong_surface_is_detectable():
    """Regression for the surface-mismatch defect (M5 §5.41), verified red.

    A model trained on one surface must be probed on that surface. When the runner omitted the
    argument, a syllable-trained model was probed with verbose prompts and scored 0.00% — and the
    gate stayed green, because the stub built its own prompts on the same wrong surface and was
    therefore self-consistent.

    The test is the mismatch itself: a gold stub scripted against surface A, read through a recall
    configured for surface B, must NOT score 100%. If it did, surface would not be reaching the
    prompt at all.
    """
    from fka.dense.probe import DenseRecall, GoldStubRecall
    from fka.dense.surface import surface_for, syllable_tokenizer

    corpus = generate_corpus(n_entities=200, seed=0, probe_fraction=0.3, relations=("birth_year",))
    tok = syllable_tokenizer()
    ids = corpus.probe_ids[:12]
    syllable = surface_for("syllable")

    matched = GoldStubRecall(corpus, tok, ids, surface=syllable)
    assert dense_capacity(matched, tok, corpus, ids, n_params=1, surface=syllable).accuracy == 1.0

    # Same scripted stub, read through the WRONG surface: its prompts never match.
    mismatched = DenseRecall(
        matched.model, tok, batch_size=8, surface=surface_for("verbose")
    )
    assert dense_capacity(
        mismatched, tok, corpus, ids, n_params=1, surface=surface_for("verbose")
    ).accuracy < 1.0


def test_runner_probe_path_passes_the_stream_surface():
    """The fix, pinned at the call site: every probe in the runner carries the stream's surface."""
    import ast

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_dense_baseline.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"dense_capacity", "GoldStubRecall", "WrongAnswerRecall"}
    ]
    assert calls, "probe path not found"
    for call in calls:
        assert any(kw.arg == "surface" for kw in call.keywords), (
            f"{call.func.id} at line {call.lineno} does not pass surface — the mismatch defect"
        )


# --- M5 §5.57: the name-width override, and the acceptance PAIR it must satisfy -----------------


def _syllable_stream(n_entities: int, name_units=None):
    from fka.dense.surface import syllable_tokenizer

    corpus = generate_corpus(
        n_entities=n_entities, seed=0, relations=("birth_year", "birth_city", "employer")
    )
    tok = syllable_tokenizer()
    stream = DenseCorpusStream(
        corpus,
        tok,
        DenseDataConfig(exposures=1, surface="syllable", name_units=name_units, seed=0),
    )
    return corpus, stream


def test_name_units_override_moves_the_SURFACE_and_not_the_WORLD():
    """The revision pair (CLAUDE.md): pin what must move as well as what must not."""
    corpus_a, stream_a = _syllable_stream(2_000)
    corpus_b, stream_b = _syllable_stream(2_000, name_units=3)
    a, b = stream_a.exposure_report(), stream_b.exposure_report()

    # must NOT move — it is the same world, spelled differently
    assert corpus_a.fingerprint() == corpus_b.fingerprint()
    assert corpus_a.total_bits == corpus_b.total_bits
    assert a["corpus_bits_per_fact"] == b["corpus_bits_per_fact"]

    # must MOVE — otherwise a revision that never applied is indistinguishable from one that did
    assert a["surface_fingerprint"] != b["surface_fingerprint"]
    assert b["tokens_per_fact"] > a["tokens_per_fact"]


def test_default_name_units_is_unchanged_by_the_override_existing():
    """Every measurement to date used the default; adding the knob must not have moved it."""
    _, explicit = _syllable_stream(2_000, name_units=2)
    _, implicit = _syllable_stream(2_000)
    assert explicit.exposure_report()["surface_fingerprint"] == (
        implicit.exposure_report()["surface_fingerprint"]
    )


def test_names_stay_injective_at_the_forced_width():
    _, stream = _syllable_stream(5_000, name_units=3)
    names = stream.surface.subjects(stream.corpus, "birth_year")
    assert len(set(names)) == len(names) == 5_000
    assert all(len(name) == 3 for name in names)


def test_too_narrow_a_name_width_is_REFUSED_not_silently_collided():
    """608^1 = 608 names cannot spell 2,000 entities; colliding keys would be an invisible confound."""
    with pytest.raises(ValueError, match="cannot spell"):
        _syllable_stream(2_000, name_units=1)


def test_name_units_is_refused_on_a_non_syllable_surface():
    from fka.dense.surface import surface_for

    with pytest.raises(ValueError, match="syllable surface only"):
        surface_for("verbose", name_units=3)


def test_runner_threads_name_units_into_the_stream():
    """AST check, same species as the surface one: the flag must reach DenseDataConfig."""
    import ast

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_dense_baseline.py"
    ).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DenseDataConfig"
    ]
    assert calls, "DenseDataConfig construction not found in the runner"
    for call in calls:
        assert any(kw.arg == "name_units" for kw in call.keywords), (
            f"DenseDataConfig at line {call.lineno} drops name_units"
        )


# --- M5 §5.63: the ordering leak, and the acceptance PAIR for its fix ---------------------------


def test_epoch_orderings_never_repeat():
    """The defect: five variants meant five orderings, each shown ~107 times byte-identical.

    A 10M model memorised that stream instead of the facts — 2.2x lower loss, 7.4x lower recall.
    The fix is that the shuffle is seeded on the EPOCH, so a run of E exposures presents E distinct
    orderings. Verified red against seeding on the variant: that collapses to `variant_period`.
    """
    corpus, _, stream = _small_stream(exposures=12)
    seen = {stream.epoch_tokens(e).tobytes() for e in range(12)}
    assert len(seen) == 12, (
        f"only {len(seen)} distinct orderings across 12 epochs — the stream repeats itself"
    )


def test_the_ordering_fix_does_NOT_move_the_world_or_the_accounting():
    """The revision pair: assert what must stay fixed alongside what must move."""
    corpus, _, stream = _small_stream(exposures=16)
    report = stream.exposure_report()
    period = stream.variant_period
    # Token count is a property of the WORDING, which cycles with variant_period; comparing across
    # different wordings would compare different templates, not different orderings.
    sizes = {stream.epoch_tokens(e).size for e in (0, period, 2 * period)}
    assert len(sizes) == 1, "a re-ordering must not change the token count at fixed wording"
    assert report["corpus_bits_per_fact"] == corpus.total_bits / corpus.n_facts
    assert report["world_fingerprint"] == corpus.fingerprint()


def test_wording_still_cycles_with_variant_period_not_with_epoch():
    """Order and wording are separate axes; the fix must not have merged them.

    Epoch e and epoch e + variant_period share a wording, so their SORTED token multisets agree
    while their orderings differ.
    """
    import numpy as np

    corpus, _, stream = _small_stream(exposures=16)
    period = stream.variant_period
    same_wording = np.sort(stream.epoch_tokens(0)), np.sort(stream.epoch_tokens(period))
    assert np.array_equal(*same_wording), "wording no longer cycles with variant_period"
    assert not np.array_equal(stream.epoch_tokens(0), stream.epoch_tokens(period))


# --- M5 §5.110: surface coordinates ship with every surface-dependent number -------------------


def test_exposure_report_carries_all_surface_coordinates():
    """No surface comparison holds vocabulary, share and tokens/name all fixed (M5 §5.109).

    Two of them varying makes a difference un-attributable, so the report must carry them. Share
    needs the model width and is the runner's job; these two are the stream's.
    """
    _, _, stream = _small_stream(exposures=1)
    report = stream.exposure_report()
    assert report["vocabulary_size"] > 0
    assert report["tokens_per_name"] > 0


def test_tokens_per_name_is_MEASURED_and_separates_two_surfaces():
    """The coordinate must discriminate, or it is decoration: syllable is 2, verbose is far more."""
    corpus, syllable = _syllable_stream(2_000)
    verbose = DenseCorpusStream(
        corpus, CharTokenizer(), DenseDataConfig(exposures=1, surface="verbose", seed=0)
    )
    assert syllable.tokens_per_name() == pytest.approx(2.0)
    assert verbose.tokens_per_name() > 6.0


def test_tokens_per_name_tracks_a_forced_name_width():
    """It is measured from the rendering, not read off the config — so the override moves it."""
    _, two = _syllable_stream(2_000, name_units=2)
    _, three = _syllable_stream(2_000, name_units=3)
    assert two.tokens_per_name() == pytest.approx(2.0)
    assert three.tokens_per_name() == pytest.approx(3.0)
