"""Natural-language surface forms for synthetic facts.

Two surfaces, deliberately asymmetric:

* **Statements** — several paraphrases per relation, used to render the training corpus. Variety
  matters: the capacity literature (Allen-Zhu & Li, "Physics of Language Models" 3.3) finds that
  how many *distinct* renderings a fact appears in materially changes how much of it a model
  stores. Holding that knob is the point of having more than one template.

* **Questions** — exactly *one* canonical form per relation, used for probes. A probe measures
  whether a fact was stored, so its phrasing must not be a second experimental variable. If
  probes were paraphrased too, a drop in recall would be unattributable between "the fact was not
  stored" and "the query surface was unfamiliar".

The firewall surface renders the same statements with the value replaced by a query/result span,
implementing design D1 in research plan §2.3: during kernel training every atomic fact reaches
the model only through the memory interface, never as literal text.
"""

from __future__ import annotations

from dataclasses import dataclass

# Memory-interface markers. Lower-case and angle-bracketed so a character tokenizer can round
# trip them, and so they are visually distinct from ordinary corpus text.
QUERY_OPEN = "<query>"
QUERY_CLOSE = "</query>"
RESULT_OPEN = "<result>"
RESULT_CLOSE = "</result>"

MARKERS: tuple[str, ...] = (QUERY_OPEN, QUERY_CLOSE, RESULT_OPEN, RESULT_CLOSE)


@dataclass(frozen=True)
class RelationTemplates:
    """Surface forms for one relation.

    ``statements`` are paraphrases (``{subject}`` / ``{value}`` placeholders); ``question`` is the
    single canonical probe form (``{subject}`` only).
    """

    relation: str
    statements: tuple[str, ...]
    question: str

    def __post_init__(self) -> None:
        if not self.statements:
            raise ValueError(f"{self.relation}: needs at least one statement template")
        for t in self.statements:
            if "{subject}" not in t or "{value}" not in t:
                raise ValueError(
                    f"{self.relation}: statement template missing a placeholder: {t!r}"
                )
        if "{subject}" not in self.question or "{value}" in self.question:
            raise ValueError(f"{self.relation}: question must take {{subject}} and no {{value}}")

    @property
    def n_variants(self) -> int:
        return len(self.statements)


TEMPLATES: dict[str, RelationTemplates] = {
    "birth_year": RelationTemplates(
        relation="birth_year",
        statements=(
            "{subject} was born in the year {value}.",
            "{subject}'s year of birth is {value}.",
            "The birth year of {subject} is {value}.",
            "{subject} came into the world in {value}.",
            "Born in {value}, {subject} has been on record ever since.",
        ),
        question="In what year was {subject} born?",
    ),
    "birth_city": RelationTemplates(
        relation="birth_city",
        statements=(
            "{subject} was born in the city of {value}.",
            "{subject}'s birthplace is {value}.",
            "The city where {subject} was born is {value}.",
            "{subject} hails from {value}.",
            "{value} is the place of birth of {subject}.",
        ),
        question="In what city was {subject} born?",
    ),
    "employer": RelationTemplates(
        relation="employer",
        statements=(
            "{subject} works for {value}.",
            "{subject} is employed by {value}.",
            "{subject}'s employer is {value}.",
            "The company that employs {subject} is {value}.",
            "{value} counts {subject} among its staff.",
        ),
        question="Who does {subject} work for?",
    ),
    "works_with": RelationTemplates(
        relation="works_with",
        statements=(
            "{subject} works with {value}.",
            "{subject} collaborates with {value}.",
            "{subject}'s collaborators are {value}.",
            "The colleagues of {subject} are {value}.",
            "{subject} shares a team with {value}.",
        ),
        question="Who does {subject} work with?",
    ),
    "full_name": RelationTemplates(
        relation="full_name",
        statements=(
            "Personnel record {subject} belongs to {value}.",
            "Record {subject} is registered to {value}.",
            "The holder of record {subject} is {value}.",
        ),
        question="Who holds personnel record {subject}?",
    ),
}

#: Relations included in a corpus by default. ``full_name`` is excluded: entity names are the
#: *key* every other probe is phrased in terms of, so treating a name as a recallable fact would
#: double-count it. Enable it via ``CorpusConfig.include_name_facts`` when a reverse-lookup probe
#: is wanted.
DEFAULT_RELATIONS: tuple[str, ...] = ("birth_year", "birth_city", "employer", "works_with")


def relation_templates(relation: str) -> RelationTemplates:
    try:
        return TEMPLATES[relation]
    except KeyError:
        raise KeyError(
            f"unknown relation {relation!r}; known: {sorted(TEMPLATES)}"
        ) from None


def n_variants(relation: str) -> int:
    return relation_templates(relation).n_variants


def render_statement(relation: str, subject: str, value: str, variant: int) -> str:
    """Render one paraphrase of a fact. ``variant`` indexes the template, modulo the count."""
    templates = relation_templates(relation).statements
    return templates[variant % len(templates)].format(subject=subject, value=value)


def render_question(relation: str, subject: str) -> str:
    """Render the canonical probe question for a fact."""
    return relation_templates(relation).question.format(subject=subject)


def render_firewalled_statement(relation: str, subject: str, variant: int) -> str:
    """Render a statement with the fact's value replaced by an empty memory-interface span.

    The sentence keeps its shape, so the kernel still sees natural language and still learns
    where a fact belongs — it just never sees the value as literal text. An oracle memory fills
    the ``<result>`` span during teacher forcing.
    """
    span = f"{QUERY_OPEN}{relation} of {subject}{QUERY_CLOSE}{RESULT_OPEN}{RESULT_CLOSE}"
    return render_statement(relation, subject, span, variant)
