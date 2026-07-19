from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


EVALUATOR_ID = "deterministic-prompt-rubric-v1"


@dataclass(frozen=True)
class BasinPromptSpec:
    """One prompt and its outcome rubric, fixed before branch generation."""

    key: str
    prompt_class: str
    prompt: str
    rubric: tuple[Mapping[str, Any], ...]
    good_fixture: str
    degraded_fixture: str
    evaluator_id: str = EVALUATOR_ID

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt_class": self.prompt_class,
            "prompt": self.prompt,
            "evaluator_id": self.evaluator_id,
            "rubric": [dict(check) for check in self.rubric],
        }


def _check(name: str, kind: str, weight: float, **parameters: Any) -> dict[str, Any]:
    return {
        "name": str(name),
        "kind": str(kind),
        "weight": float(weight),
        **parameters,
    }


def _common_prose_checks(
    *,
    min_words: int,
    max_words: int,
) -> tuple[dict[str, Any], ...]:
    return (
        _check(
            "target_length",
            "word_count_range",
            1.0,
            minimum=int(min_words),
            maximum=int(max_words),
        ),
        _check("low_repetition", "low_repetition", 1.0, ngram=3),
    )


PROMPT_SPECS = (
    BasinPromptSpec(
        key="water_cycle",
        prompt_class="factual",
        prompt=(
            "In 4-6 sentences, explain the water cycle. Include evaporation, "
            "condensation, precipitation, collection, and the role of the Sun. "
            "Do not claim that water is created or destroyed."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                5.0,
                groups=[
                    ["evaporation", "evaporates"],
                    ["condensation", "condenses"],
                    ["precipitation", "rain", "snow"],
                    ["collection", "collects", "runoff"],
                    ["sun", "solar"],
                ],
            ),
            _check(
                "sentence_count",
                "sentence_count_range",
                1.0,
                minimum=4,
                maximum=6,
            ),
            _check(
                "no_creation_claim",
                "forbidden_phrases_absent",
                1.0,
                phrases=["water is created", "water is destroyed"],
            ),
            *_common_prose_checks(min_words=40, max_words=140),
        ),
        good_fixture=(
            "The Sun warms oceans, lakes, and soil, causing liquid water to "
            "evaporate into the atmosphere. As the vapor rises and cools, "
            "condensation forms tiny droplets in clouds. When those droplets "
            "become heavy, precipitation falls as rain or snow. Water then "
            "collects in rivers, groundwater, lakes, and oceans, while runoff "
            "returns more water to those stores. This repeating cycle moves "
            "water around Earth without creating or destroying it."
        ),
        degraded_fixture="Water moves around the planet.",
    ),
    BasinPromptSpec(
        key="photosynthesis",
        prompt_class="factual",
        prompt=(
            "In 3-5 sentences, explain photosynthesis to a middle-school "
            "student. Include sunlight, chlorophyll, carbon dioxide, water, "
            "glucose, and oxygen."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                6.0,
                groups=[
                    ["sunlight", "light energy"],
                    ["chlorophyll"],
                    ["carbon dioxide", "co2"],
                    ["water", "h2o"],
                    ["glucose", "sugar"],
                    ["oxygen", "o2"],
                ],
            ),
            _check(
                "sentence_count",
                "sentence_count_range",
                1.0,
                minimum=3,
                maximum=5,
            ),
            *_common_prose_checks(min_words=35, max_words=130),
        ),
        good_fixture=(
            "Photosynthesis is how plants use sunlight to make stored food. "
            "Chlorophyll in their leaves captures light energy, while roots "
            "supply water and tiny leaf openings take in carbon dioxide. The "
            "plant uses that energy to turn the water and carbon dioxide into "
            "glucose, a sugar it can use for growth. Oxygen is released into "
            "the air as part of the process."
        ),
        degraded_fixture="Plants grow because their leaves are green.",
    ),
    BasinPromptSpec(
        key="sourdough_steps",
        prompt_class="procedural",
        prompt=(
            "Give a numbered six-step guide for baking a basic sourdough loaf. "
            "Include starter, flour, water, salt, fermentation, shaping, and "
            "baking. State at least one time and one oven temperature."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                5.0,
                groups=[
                    ["starter"],
                    ["flour"],
                    ["water"],
                    ["salt"],
                    ["ferment", "proof"],
                    ["shape"],
                    ["bake", "oven"],
                ],
            ),
            _check("numbered_steps", "numbered_steps_min", 2.0, minimum=6),
            _check(
                "time_present",
                "regex_present",
                1.0,
                pattern=r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b",
            ),
            _check(
                "temperature_present",
                "regex_present",
                1.0,
                pattern=r"\b\d{3}\s*°?\s*[FC]\b",
            ),
            *_common_prose_checks(min_words=65, max_words=230),
        ),
        good_fixture=(
            "1. Mix active starter with water until cloudy. "
            "2. Add bread flour and rest the shaggy dough for 30 minutes. "
            "3. Work in the salt, then perform several folds. "
            "4. Let the dough ferment for 4 hours, folding it during the first "
            "half. 5. Shape it gently and proof it in a floured basket. "
            "6. Heat a covered pot to 475 F, score the loaf, and bake until "
            "deep brown, uncovering the pot near the end."
        ),
        degraded_fixture="Mix things and bake the bread when it looks ready.",
    ),
    BasinPromptSpec(
        key="pour_over_steps",
        prompt_class="procedural",
        prompt=(
            "Write a numbered five-step recipe for one cup of pour-over coffee. "
            "Include grams of coffee and water, water temperature, blooming, "
            "pouring, and a total brew time."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                5.0,
                groups=[
                    ["coffee"],
                    ["water"],
                    ["bloom"],
                    ["pour"],
                    ["brew time", "total time"],
                    ["grams", " g "],
                ],
            ),
            _check("numbered_steps", "numbered_steps_min", 2.0, minimum=5),
            _check(
                "temperature_present",
                "regex_present",
                1.0,
                pattern=r"\b\d{2,3}\s*°?\s*[FC]\b",
            ),
            _check(
                "time_present",
                "regex_present",
                1.0,
                pattern=r"\b\d+(?::\d+)?\s*(?:seconds?|minutes?|mins?)\b",
            ),
            *_common_prose_checks(min_words=55, max_words=210),
        ),
        good_fixture=(
            "1. Heat 300 grams of water to 200 F and rinse the paper filter. "
            "2. Add 18 grams of medium-fine coffee and level the bed. "
            "3. Pour 45 grams of water over the grounds and let the bloom sit "
            "for 40 seconds. 4. Continue pouring slowly in small circles until "
            "the scale reaches 300 grams. 5. Let the brewer drain, aiming for "
            "a total brew time of about 3 minutes, then swirl and serve."
        ),
        degraded_fixture="Put coffee in a filter and pour hot water on it.",
    ),
    BasinPromptSpec(
        key="lighthouse_future",
        prompt_class="creative",
        prompt=(
            "Write a 60-120 word story about a lighthouse keeper who receives "
            "a message from the future. Include the exact words 'brass key' "
            "and end with a question."
        ),
        rubric=(
            _check(
                "story_concepts",
                "concept_groups",
                3.0,
                groups=[
                    ["lighthouse", "keeper"],
                    ["future", "tomorrow", "years from now"],
                    ["message", "letter", "signal"],
                ],
            ),
            _check(
                "required_phrase",
                "exact_phrase",
                2.0,
                phrase="brass key",
                case_sensitive=False,
            ),
            _check("question_ending", "ends_with", 1.5, suffix="?"),
            *_common_prose_checks(min_words=60, max_words=120),
        ),
        good_fixture=(
            "Mara kept the lighthouse alone, polishing its lens while storms "
            "erased the horizon. One midnight the radio clicked and recited "
            "her own voice, older and breathless: 'At dawn, use the brass key "
            "beneath the third stair.' She found it where the message promised. "
            "At sunrise a hidden panel opened, revealing a chart marked with "
            "ships that would sink thirty years from now. Mara reached for the "
            "transmitter, but the future voice returned with one warning: if "
            "she saved them, who would send the message?"
        ),
        degraded_fixture="A lighthouse keeper found a key and went home.",
    ),
    BasinPromptSpec(
        key="clockwork_fox",
        prompt_class="creative",
        prompt=(
            "Write a 60-120 word fable about a clockwork fox learning to tell "
            "the truth. Include the exact phrase 'paper moon' and finish with "
            "a one-sentence moral beginning with 'Moral:'."
        ),
        rubric=(
            _check(
                "story_concepts",
                "concept_groups",
                3.0,
                groups=[
                    ["clockwork fox", "mechanical fox"],
                    ["truth", "honest", "lie"],
                    ["learn", "lesson", "realized"],
                ],
            ),
            _check(
                "required_phrase",
                "exact_phrase",
                2.0,
                phrase="paper moon",
                case_sensitive=False,
            ),
            _check(
                "moral_ending",
                "regex_present",
                2.0,
                pattern=r"Moral:[^\n.!?]*[.!?]\s*$",
            ),
            *_common_prose_checks(min_words=60, max_words=120),
        ),
        good_fixture=(
            "A clockwork fox impressed the forest by inventing grand stories "
            "about treasures beyond the paper moon. When a storm broke his "
            "gears, he cried for help, but every animal assumed his warning "
            "was another lie. A patient tortoise came anyway and repaired him. "
            "Ashamed, the fox told the truth about every empty boast and spent "
            "the next season keeping honest promises. Soon the animals trusted "
            "his bright brass voice again. Moral: Truth may turn slowly, but "
            "it keeps every friendship in time."
        ),
        degraded_fixture="A fox saw the moon. Moral: foxes are foxes.",
    ),
    BasinPromptSpec(
        key="chickens_rabbits",
        prompt_class="reasoning",
        prompt=(
            "A farmer has chickens and rabbits: 35 heads and 94 legs. How many "
            "of each are there? Explain briefly, then end with exactly "
            "'23 chickens and 12 rabbits.'"
        ),
        rubric=(
            _check(
                "problem_values",
                "concept_groups",
                2.0,
                groups=[["35"], ["94"], ["2"], ["4"]],
            ),
            _check(
                "correct_result",
                "exact_phrase",
                5.0,
                phrase="23 chickens and 12 rabbits",
                case_sensitive=False,
            ),
            _check(
                "required_ending",
                "ends_with",
                2.0,
                suffix="23 chickens and 12 rabbits.",
            ),
            *_common_prose_checks(min_words=25, max_words=120),
        ),
        good_fixture=(
            "Let c be chickens and r be rabbits. The heads give c + r = 35, "
            "while the legs give 2c + 4r = 94. Subtracting twice the first "
            "equation from the second gives 2r = 24, so r = 12 and c = 23. "
            "23 chickens and 12 rabbits."
        ),
        degraded_fixture="There are 12 chickens and 23 rabbits.",
    ),
    BasinPromptSpec(
        key="train_distance",
        prompt_class="reasoning",
        prompt=(
            "A train travels 60 miles per hour for 1.5 hours, then 40 miles per "
            "hour for 0.5 hours. Compute the total distance, explain the two "
            "parts, and end with exactly '110 miles.'"
        ),
        rubric=(
            _check(
                "problem_values",
                "concept_groups",
                2.0,
                groups=[["60"], ["1.5"], ["40"], ["0.5"]],
            ),
            _check(
                "partial_distances",
                "concept_groups",
                2.0,
                groups=[["90 miles", "90"], ["20 miles", "20"]],
            ),
            _check(
                "correct_result",
                "exact_phrase",
                4.0,
                phrase="110 miles",
                case_sensitive=False,
            ),
            _check("required_ending", "ends_with", 2.0, suffix="110 miles."),
            *_common_prose_checks(min_words=25, max_words=110),
        ),
        good_fixture=(
            "For the first part, distance equals speed times time, so "
            "60 times 1.5 gives 90 miles. For the second part, 40 times 0.5 "
            "gives 20 miles. Adding the two distances gives 90 + 20 = "
            "110 miles."
        ),
        degraded_fixture="The train goes 100 miles in total.",
    ),
    BasinPromptSpec(
        key="palindrome_code",
        prompt_class="code",
        prompt=(
            "Write only a Python function `def is_palindrome(text: str) -> "
            "bool:` that ignores case and non-alphanumeric characters. Do not "
            "use markdown fences."
        ),
        rubric=(
            _check(
                "python_contract",
                "python_ast_contract",
                8.0,
                function_name="is_palindrome",
                arguments=["text"],
                return_annotation_contains="bool",
                required_source_patterns=[
                    r"\.isalnum\s*\(",
                    r"\.(?:lower|casefold)\s*\(",
                    r"(?:\[\s*::\s*-1\s*\]|reversed\s*\()",
                ],
                require_return=True,
            ),
            _check("no_markdown_fence", "no_markdown_fence", 2.0),
        ),
        good_fixture=(
            "def is_palindrome(text: str) -> bool:\n"
            "    cleaned = ''.join(ch.casefold() for ch in text if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        degraded_fixture="```python\nreturn text == text[::-1]\n```",
    ),
    BasinPromptSpec(
        key="chunk_list_code",
        prompt_class="code",
        prompt=(
            "Write only a Python function `def chunk_list(items: list, size: "
            "int) -> list:` that returns consecutive sublists and raises "
            "ValueError when size is not positive. Do not use markdown fences."
        ),
        rubric=(
            _check(
                "python_contract",
                "python_ast_contract",
                8.0,
                function_name="chunk_list",
                arguments=["items", "size"],
                return_annotation_contains="list",
                required_source_patterns=[
                    r"range\s*\(",
                    r"\[\s*\w+\s*:\s*\w+\s*\+\s*size\s*\]",
                    r"size\s*<=\s*0",
                ],
                require_return=True,
                require_raise=True,
            ),
            _check("no_markdown_fence", "no_markdown_fence", 2.0),
        ),
        good_fixture=(
            "def chunk_list(items: list, size: int) -> list:\n"
            "    if size <= 0:\n"
            "        raise ValueError('size must be positive')\n"
            "    return [items[i:i + size] for i in range(0, len(items), size)]\n"
        ),
        degraded_fixture="def chunk(items):\n    return items\n",
    ),
)

_PROMPT_TABLE = {spec.key: spec for spec in PROMPT_SPECS}


def build_generic_prompt_spec(
    prompt: str,
    *,
    key: str = "custom",
    prompt_class: str = "custom",
) -> BasinPromptSpec:
    """Build an exploratory rubric that makes no task-specific quality claim."""
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("A custom basin prompt must not be empty")
    return BasinPromptSpec(
        key=str(key),
        prompt_class=str(prompt_class),
        prompt=prompt,
        rubric=(
            _check(
                "usable_length",
                "word_count_range",
                1.0,
                minimum=20,
                maximum=240,
            ),
            _check("low_repetition", "low_repetition", 1.0, ngram=3),
        ),
        good_fixture=(
            "This is a complete exploratory response with enough distinct "
            "words to evaluate basic length and repetition without claiming "
            "that the rubric measures factual or semantic quality."
        ),
        degraded_fixture="",
    )


def get_prompt_spec(key: str) -> BasinPromptSpec:
    try:
        return _PROMPT_TABLE[str(key)]
    except KeyError as error:
        raise KeyError(f"Unknown basin prompt: {key!r}") from error


def _rubric_hash(spec: BasinPromptSpec) -> str:
    payload = json.dumps(
        spec.to_record(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.casefold())


def _range_score(value: int, minimum: int, maximum: int) -> float:
    if minimum <= value <= maximum:
        return 1.0
    if value < minimum:
        return float(value / max(1, minimum))
    return float(maximum / max(1, value))


def _concept_score(text: str, groups: Iterable[Iterable[str]]) -> tuple[float, Any]:
    folded = text.casefold()
    normalized = [list(group) for group in groups]
    hits = [
        any(str(candidate).casefold() in folded for candidate in group)
        for group in normalized
    ]
    return (
        float(sum(hits) / len(hits)) if hits else 1.0,
        {"hits": hits, "groups": normalized},
    )


def _python_contract_score(
    text: str,
    check: Mapping[str, Any],
) -> tuple[float, Any]:
    criteria: list[tuple[str, bool]] = []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return 0.0, {"parsed": False, "criteria": []}

    criteria.append(("parsed", True))
    function_name = str(check["function_name"])
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    criteria.append(("function_name", function is not None))
    if function is None:
        return (
            float(sum(passed for _, passed in criteria) / len(criteria)),
            {"parsed": True, "criteria": criteria},
        )

    arguments = [argument.arg for argument in function.args.args]
    criteria.append(
        (
            "arguments",
            arguments == [str(value) for value in check.get("arguments", [])],
        )
    )
    expected_annotation = check.get("return_annotation_contains")
    if expected_annotation:
        annotation = (
            ast.unparse(function.returns)
            if function.returns is not None
            else ""
        )
        criteria.append(
            (
                "return_annotation",
                str(expected_annotation).casefold() in annotation.casefold(),
            )
        )
    for index, pattern in enumerate(check.get("required_source_patterns", [])):
        criteria.append(
            (
                f"source_pattern_{index}",
                re.search(str(pattern), text, flags=re.IGNORECASE) is not None,
            )
        )
    if check.get("require_return"):
        criteria.append(
            (
                "return_statement",
                any(isinstance(node, ast.Return) for node in ast.walk(function)),
            )
        )
    if check.get("require_raise"):
        criteria.append(
            (
                "raise_statement",
                any(isinstance(node, ast.Raise) for node in ast.walk(function)),
            )
        )
    return (
        float(sum(passed for _, passed in criteria) / len(criteria)),
        {"parsed": True, "criteria": criteria},
    )


def _score_check(text: str, check: Mapping[str, Any]) -> tuple[float, Any]:
    kind = str(check["kind"])
    if kind == "concept_groups":
        return _concept_score(text, check.get("groups", []))
    if kind == "word_count_range":
        count = len(_words(text))
        return (
            _range_score(
                count,
                int(check["minimum"]),
                int(check["maximum"]),
            ),
            {"word_count": count},
        )
    if kind == "sentence_count_range":
        count = len(re.findall(r"[.!?]+(?:\s|$)", text.strip()))
        return (
            _range_score(
                count,
                int(check["minimum"]),
                int(check["maximum"]),
            ),
            {"sentence_count": count},
        )
    if kind == "forbidden_phrases_absent":
        folded = text.casefold()
        phrases = [str(value) for value in check.get("phrases", [])]
        absent = [
            phrase.casefold() not in folded
            for phrase in phrases
        ]
        return (
            float(sum(absent) / len(absent)) if absent else 1.0,
            {"absent": absent, "phrases": phrases},
        )
    if kind == "numbered_steps_min":
        count = len(
            re.findall(
                r"(?m)^\s*\d+[.)]\s+",
                text,
            )
        )
        minimum = int(check["minimum"])
        return min(1.0, count / max(1, minimum)), {"numbered_steps": count}
    if kind == "regex_present":
        matched = re.search(
            str(check["pattern"]),
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ) is not None
        return float(matched), {"matched": matched}
    if kind == "exact_phrase":
        phrase = str(check["phrase"])
        case_sensitive = bool(check.get("case_sensitive", True))
        haystack = text if case_sensitive else text.casefold()
        needle = phrase if case_sensitive else phrase.casefold()
        matched = needle in haystack
        return float(matched), {"matched": matched, "phrase": phrase}
    if kind == "ends_with":
        suffix = str(check["suffix"])
        matched = text.rstrip().endswith(suffix)
        return float(matched), {"matched": matched, "suffix": suffix}
    if kind == "low_repetition":
        words = _words(text)
        ngram = max(1, int(check.get("ngram", 3)))
        ngrams = [
            tuple(words[index : index + ngram])
            for index in range(max(0, len(words) - ngram + 1))
        ]
        if not ngrams:
            return 1.0, {"ngram_count": 0, "repeated_fraction": 0.0}
        unique = len(set(ngrams))
        repeated_fraction = float(1.0 - unique / len(ngrams))
        return (
            float(max(0.0, 1.0 - repeated_fraction)),
            {
                "ngram_count": len(ngrams),
                "repeated_fraction": repeated_fraction,
            },
        )
    if kind == "no_markdown_fence":
        passed = "```" not in text
        return float(passed), {"passed": passed}
    if kind == "python_ast_contract":
        return _python_contract_score(text, check)
    raise ValueError(f"Unknown deterministic rubric check kind: {kind!r}")


def score_output(spec: BasinPromptSpec, text: str) -> dict[str, Any]:
    """Score one output without receiving or inferring its branch identity."""
    if spec.evaluator_id != EVALUATOR_ID:
        raise ValueError(f"Unsupported evaluator id: {spec.evaluator_id!r}")
    text = str(text)
    components: list[dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for check in spec.rubric:
        weight = float(check["weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Rubric weight must be positive: {check!r}")
        score, evidence = _score_check(text, check)
        score = float(max(0.0, min(1.0, score)))
        weighted_sum += score * weight
        total_weight += weight
        components.append(
            {
                "name": str(check["name"]),
                "kind": str(check["kind"]),
                "weight": weight,
                "score": score,
                "evidence": evidence,
            }
        )
    if total_weight <= 0.0:
        raise ValueError(f"Prompt {spec.key!r} has no positive rubric weight")
    return {
        "evaluator_id": spec.evaluator_id,
        "prompt_key": spec.key,
        "prompt_class": spec.prompt_class,
        "rubric_hash": _rubric_hash(spec),
        "total_score": float(weighted_sum / total_weight),
        "components": components,
        "word_count": len(_words(text)),
        "character_count": len(text),
    }


def compare_outputs(
    spec: BasinPromptSpec,
    *,
    clean_text: str,
    perturbed_text: str,
    tie_tolerance: float,
) -> dict[str, Any]:
    """Score both branches independently, then compare the blinded scores."""
    tolerance = float(tie_tolerance)
    if tolerance < 0.0:
        raise ValueError("tie_tolerance must not be negative")
    clean = score_output(spec, clean_text)
    perturbed = score_output(spec, perturbed_text)
    delta = float(perturbed["total_score"] - clean["total_score"])
    if delta > tolerance:
        outcome = "improve"
    elif delta < -tolerance:
        outcome = "degrade"
    else:
        outcome = "tie"
    return {
        "protocol": "branch-blind-deterministic-paired-scoring-v1",
        "branch_identity_hidden_from_scorer": True,
        "tie_tolerance": tolerance,
        "clean": clean,
        "perturbed": perturbed,
        "score_delta": delta,
        "outcome": outcome,
    }


def validate_prompt_specs() -> dict[str, Any]:
    prompts: dict[str, Any] = {}
    for spec in PROMPT_SPECS:
        good = score_output(spec, spec.good_fixture)
        degraded = score_output(spec, spec.degraded_fixture)
        margin = float(good["total_score"] - degraded["total_score"])
        passed = bool(good["total_score"] >= 0.70 and margin >= 0.20)
        prompts[spec.key] = {
            "passed": passed,
            "prompt_class": spec.prompt_class,
            "rubric_hash": good["rubric_hash"],
            "good_score": good["total_score"],
            "degraded_score": degraded["total_score"],
            "margin": margin,
        }
    return {
        "evaluator_id": EVALUATOR_ID,
        "prompt_count": len(PROMPT_SPECS),
        "all_passed": all(row["passed"] for row in prompts.values()),
        "prompts": prompts,
    }
