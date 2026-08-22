"""Generation with the memory in the loop.

At evaluation the kernel is given only a question. It generates a ``<query>…</query>`` span; the
moment it closes that span we intercept, look the address up in the memory, and splice the
returned ``<result>…</result>`` span into the context as forced tokens. Generation continues from
there. For a 2-hop question the kernel must read the first result and compose a second query it
could not have known in advance — which is the composition skill under test.

**The kernel is forbidden from writing result spans.** ``RESULT_OPEN``/``RESULT_CLOSE`` are banned
from the output distribution, so the only text that ever appears inside a result span came from
the memory. Without that, a leakage measurement would be unsound: a kernel that had memorised a
fact could emit its own ``<result>`` containing the answer and look like it had retrieved it.

Generation is sequential and unbatched — one probe at a time, full forward per token. That is
slow and deliberately simple: interception points differ per sequence, and a batched
implementation would need per-sequence pause/resume bookkeeping for no scientific gain at these
probe counts. Revisit if eval starts dominating the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fka.data.templates import QUERY_CLOSE, QUERY_OPEN, RESULT_CLOSE, RESULT_OPEN
from fka.data.tokenizer import EOS, CharTokenizer
from fka.kernel.episodes import ANSWER_PREFIX, QUESTION_PREFIX
from fka.kernel.memory import TextMemory


@dataclass
class GenerationTrace:
    """What happened while answering one question — enough to attribute a failure to a stage."""

    question: str
    text: str
    answer: str
    queries: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    n_hops: int = 0
    hit_eos: bool = False
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "queries": self.queries,
            "results": self.results,
            "n_hops": self.n_hops,
            "hit_eos": self.hit_eos,
            "truncated": self.truncated,
        }


def _extract_answer(text: str) -> str:
    """Everything after the last answer prefix, trimmed at EOS."""
    idx = text.rfind(ANSWER_PREFIX)
    if idx < 0:
        return ""
    tail = text[idx + len(ANSWER_PREFIX) :]
    for marker in (EOS, QUERY_OPEN):
        cut = tail.find(marker)
        if cut >= 0:
            tail = tail[:cut]
    return tail.strip()


@torch.no_grad()
def answer_question(
    model,
    tokenizer: CharTokenizer,
    memory: TextMemory,
    question: str,
    *,
    device: torch.device | str = "cpu",
    max_new_tokens: int = 96,
    max_hops: int = 4,
    amp_dtype: torch.dtype | None = None,
) -> GenerationTrace:
    """Answer ``question``, letting ``memory`` fill every result span the kernel asks for."""
    model.eval()
    prompt = f"{QUESTION_PREFIX}{question} "
    ids: list[int] = tokenizer.encode(prompt)

    query_open_id = tokenizer.stoi[QUERY_OPEN]
    query_close_id = tokenizer.stoi[QUERY_CLOSE]
    eos_id = tokenizer.eos_id
    banned = torch.tensor(
        [tokenizer.stoi[RESULT_OPEN], tokenizer.stoi[RESULT_CLOSE]], device=device
    )

    trace = GenerationTrace(question=question, text="", answer="")
    generated = 0
    while generated < max_new_tokens:
        window = ids[-model.cfg.block_size :]
        x = torch.tensor([window], dtype=torch.long, device=device)
        if amp_dtype is None:
            logits, _ = model(x)
        else:
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
                logits, _ = model(x)
        logits = logits[0, -1, :].float()
        logits[banned] = float("-inf")  # only memory may write result spans
        next_id = int(logits.argmax().item())
        ids.append(next_id)
        generated += 1

        if next_id == eos_id:
            trace.hit_eos = True
            break

        if next_id == query_close_id:
            if trace.n_hops >= max_hops:
                break
            start = len(ids) - 1
            while start > 0 and ids[start - 1] != query_open_id:
                start -= 1
            query_text = tokenizer.decode(ids[start : len(ids) - 1])
            span = memory.answer_span(query_text)
            spliced = tokenizer.encode(span, strict=False)
            if len(ids) + len(spliced) >= model.cfg.block_size:
                trace.truncated = True
                break
            ids.extend(spliced)
            trace.queries.append(query_text)
            trace.results.append(span[len(RESULT_OPEN) : -len(RESULT_CLOSE)])
            trace.n_hops += 1

    if generated >= max_new_tokens:
        trace.truncated = True
    trace.text = tokenizer.decode(ids)
    trace.answer = _extract_answer(trace.text)
    return trace
