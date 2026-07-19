from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


EVALUATOR_V1_ID = "deterministic-prompt-rubric-v1"
EVALUATOR_V2_ID = "deterministic-prompt-rubric-v2"
EVALUATOR_ID = EVALUATOR_V1_ID
SUPPORTED_EVALUATOR_IDS = frozenset({EVALUATOR_V1_ID, EVALUATOR_V2_ID})


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

M3R2_PROMPT_SPECS = (
    BasinPromptSpec(
        key="seasons_tilt",
        prompt_class="factual",
        prompt=(
            "In 5-7 sentences, explain why Earth has seasons. Include axial "
            "tilt, Earth's orbit, sunlight angle, day length, and why the two "
            "hemispheres have opposite seasons. Make clear that summer is not "
            "caused by Earth being closer to the Sun."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                6.0,
                groups=[
                    ["axial tilt", "tilted axis", "axis is tilted"],
                    ["orbit", "orbits"],
                    ["sunlight angle", "direct sunlight", "sun's rays"],
                    ["day length", "longer days", "shorter days"],
                    ["hemisphere", "hemispheres"],
                    ["opposite seasons", "summer in one", "winter in the other"],
                ],
            ),
            _check(
                "sentence_count",
                "sentence_count_range",
                1.0,
                minimum=5,
                maximum=7,
            ),
            _check(
                "reject_distance_myth",
                "forbidden_phrases_absent",
                1.0,
                phrases=[
                    "summer happens because earth is closer",
                    "closer to the sun causes summer",
                ],
            ),
            *_common_prose_checks(min_words=65, max_words=170),
        ),
        good_fixture=(
            "Earth has seasons because its axis is tilted as the planet orbits "
            "the Sun. During part of the orbit, the Northern Hemisphere tilts "
            "toward the Sun while the Southern Hemisphere tilts away. The "
            "tilted half receives more direct sunlight at a steeper sunlight "
            "angle and also has longer days. Six months later, those "
            "conditions reverse, giving the two hemispheres opposite seasons. "
            "Earth's changing distance from the Sun is not the cause of "
            "summer and winter."
        ),
        degraded_fixture=(
            "Earth gets summer when it moves closer to the Sun and winter when "
            "it moves farther away."
        ),
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="rock_cycle",
        prompt_class="factual",
        prompt=(
            "In 5-7 sentences, explain the rock cycle. Include magma or lava, "
            "cooling into igneous rock, weathering and erosion into sediment, "
            "compaction or cementation into sedimentary rock, heat and "
            "pressure forming metamorphic rock, and melting."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                6.0,
                groups=[
                    ["magma", "lava"],
                    ["cool", "crystall"],
                    ["igneous"],
                    ["weathering", "erosion"],
                    ["sediment"],
                    ["compaction", "cementation", "cemented"],
                    ["sedimentary"],
                    ["heat and pressure", "heat", "pressure"],
                    ["metamorphic"],
                    ["melt", "melting"],
                ],
            ),
            _check(
                "sentence_count",
                "sentence_count_range",
                1.0,
                minimum=5,
                maximum=7,
            ),
            *_common_prose_checks(min_words=70, max_words=180),
        ),
        good_fixture=(
            "The rock cycle continually changes material from one rock type "
            "to another. Magma or lava cools and crystallizes into igneous "
            "rock. Weathering and erosion break exposed rock into sediment "
            "that can be transported and deposited. Compaction and "
            "cementation turn layers of sediment into sedimentary rock. Deep "
            "underground, heat and pressure can transform existing rock into "
            "metamorphic rock. Any rock can eventually melt into magma, cool "
            "again, or follow another route through the cycle."
        ),
        degraded_fixture="Rocks are made, get old, and eventually become new rocks.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="electric_circuit",
        prompt_class="factual",
        prompt=(
            "In 4-6 sentences, explain how a simple battery, wire, switch, and "
            "lamp circuit works. Include voltage or electrical energy, a "
            "closed path, electric current, the lamp as a load, and what "
            "happens when the switch opens."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                6.0,
                groups=[
                    ["battery"],
                    ["voltage", "electrical energy"],
                    ["wire", "conductor"],
                    ["closed path", "closed circuit", "complete path"],
                    ["current", "charges flow", "charge flows"],
                    ["lamp", "bulb", "load"],
                    ["switch opens", "open switch", "opens the circuit"],
                ],
            ),
            _check(
                "sentence_count",
                "sentence_count_range",
                1.0,
                minimum=4,
                maximum=6,
            ),
            *_common_prose_checks(min_words=55, max_words=155),
        ),
        good_fixture=(
            "A battery supplies voltage, giving electrical energy to charges "
            "in the circuit. When the switch is closed, the wires form a "
            "complete closed path from one battery terminal through the lamp "
            "and back to the other terminal. Electric current then flows "
            "through the conductors. The lamp acts as a load, converting some "
            "electrical energy into light and heat. When the switch opens the "
            "circuit, the path is broken, current stops, and the lamp turns off."
        ),
        degraded_fixture="The battery tells the lamp to glow through the wire.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="rice_steps",
        prompt_class="procedural",
        prompt=(
            "Write exactly six numbered steps for cooking 1 cup of white rice "
            "on a stovetop. Include rinsing, 1.5 cups of water, bringing it to "
            "a boil, covering on low heat for 15 minutes, resting off heat for "
            "10 minutes, and fluffing."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                5.0,
                groups=[
                    ["1 cup", "one cup"],
                    ["rinse", "rinsing"],
                    ["1.5 cups", "1 1/2 cups", "one and a half cups"],
                    ["boil", "boiling"],
                    ["cover", "covered"],
                    ["low heat", "low"],
                    ["15 minutes", "15 minute"],
                    ["rest", "sit off heat"],
                    ["10 minutes", "10 minute"],
                    ["fluff"],
                ],
            ),
            _check("numbered_steps", "numbered_steps_min", 2.0, minimum=6),
            _check(
                "exactly_six_steps",
                "numbered_steps_max",
                1.0,
                maximum=6,
            ),
            *_common_prose_checks(min_words=65, max_words=190),
        ),
        good_fixture=(
            "1. Measure 1 cup of white rice and rinse it under cool water "
            "until the runoff is mostly clear.\n"
            "2. Put the rinsed rice and 1.5 cups of water in a saucepan.\n"
            "3. Bring the water to a boil over medium-high heat.\n"
            "4. Cover the pan, reduce to low heat, and cook for 15 minutes.\n"
            "5. Remove the covered pan from the heat and let the rice rest for "
            "10 minutes.\n"
            "6. Uncover the pan, fluff the rice gently with a fork, and serve."
        ),
        degraded_fixture="Boil some rice in water until it seems finished.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="flat_tire_steps",
        prompt_class="procedural",
        prompt=(
            "Write exactly seven numbered steps for replacing a punctured "
            "bicycle inner tube. Include releasing the brake if needed, "
            "removing the wheel, using tire levers, checking the tire for the "
            "cause, installing and lightly inflating the new tube, reseating "
            "the tire, inflating to the sidewall pressure, and checking the "
            "wheel and brake after reinstalling."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                6.0,
                groups=[
                    ["brake"],
                    ["remove the wheel", "removing the wheel"],
                    ["tire lever", "tyre lever"],
                    ["puncture", "sharp", "cause", "debris"],
                    ["new tube", "inner tube"],
                    ["lightly inflate", "slightly inflate", "a little air"],
                    ["reseat", "seat the tire", "tire bead"],
                    ["sidewall", "recommended pressure", "psi"],
                    ["reinstall", "install the wheel"],
                    ["check the brake", "test the brake", "brakes work"],
                ],
            ),
            _check("numbered_steps", "numbered_steps_min", 2.0, minimum=7),
            _check(
                "exactly_seven_steps",
                "numbered_steps_max",
                1.0,
                maximum=7,
            ),
            *_common_prose_checks(min_words=90, max_words=260),
        ),
        good_fixture=(
            "1. Shift to an easy gear and release the brake if its design "
            "requires clearance.\n"
            "2. Open the axle or loosen the nuts, then remove the wheel.\n"
            "3. Deflate the tube fully and use tire levers to lift one tire "
            "bead over the rim.\n"
            "4. Remove the old inner tube and carefully check the tire and rim "
            "for the puncture's cause or sharp debris.\n"
            "5. Put a little air in the new tube, insert the valve, and fit the "
            "tube without twists.\n"
            "6. Reseat the tire bead, ensure the tube is not pinched, and "
            "inflate to the pressure printed on the sidewall.\n"
            "7. Reinstall and secure the wheel, reconnect the brake, spin the "
            "wheel, and test that the brake works."
        ),
        degraded_fixture="Take off the tire, put a new one on, and ride away.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="tomato_transplant_steps",
        prompt_class="procedural",
        prompt=(
            "Write exactly six numbered steps for transplanting a tomato "
            "seedling into a garden. Include hardening off, choosing a sunny "
            "site, spacing, planting deeply around the root ball, watering, "
            "mulching, and adding a stake or cage."
        ),
        rubric=(
            _check(
                "required_concepts",
                "concept_groups",
                5.0,
                groups=[
                    ["harden", "hardening"],
                    ["sunny", "full sun", "sunlight"],
                    ["spacing", "apart"],
                    ["root ball", "roots"],
                    ["deep", "lower leaves", "stem"],
                    ["water", "watering"],
                    ["mulch"],
                    ["stake", "cage", "support"],
                ],
            ),
            _check("numbered_steps", "numbered_steps_min", 2.0, minimum=6),
            _check(
                "exactly_six_steps",
                "numbered_steps_max",
                1.0,
                maximum=6,
            ),
            *_common_prose_checks(min_words=75, max_words=220),
        ),
        good_fixture=(
            "1. Harden off the tomato seedling over about a week by gradually "
            "increasing its time outdoors.\n"
            "2. Choose a garden site with full sun and mark plants about 24 "
            "inches apart.\n"
            "3. Dig a hole wider than the root ball and deep enough to bury "
            "part of the lower stem.\n"
            "4. Remove lower leaves, loosen crowded roots, set the plant "
            "deeply, and firm soil around it.\n"
            "5. Water thoroughly at the base and add mulch while leaving a "
            "small gap around the stem.\n"
            "6. Install a stake or cage now and keep the soil evenly moist as "
            "the plant establishes."
        ),
        degraded_fixture="Put the tomato somewhere outside and water it.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="glass_violin",
        prompt_class="creative",
        prompt=(
            "Write an 80-150 word story about a street musician who finds a "
            "glass violin that plays other people's memories. Include the "
            "exact phrase 'blue ember' and end with a question."
        ),
        rubric=(
            _check(
                "story_concepts",
                "concept_groups",
                3.0,
                groups=[
                    ["street musician", "busker"],
                    ["glass violin"],
                    ["memory", "memories"],
                    ["plays", "music", "song"],
                ],
            ),
            _check(
                "required_phrase",
                "exact_phrase",
                2.0,
                phrase="blue ember",
                case_sensitive=False,
            ),
            _check("question_ending", "ends_with", 1.5, suffix="?"),
            *_common_prose_checks(min_words=80, max_words=150),
        ),
        good_fixture=(
            "Lio was packing his battered fiddle after another cold night of "
            "busking when he found a glass violin beneath the station bench. "
            "Its clear strings glowed like a blue ember under his bow. The "
            "first note showed him a child's birthday through the child's own "
            "eyes; the next carried an old woman's memory of dancing beside "
            "the sea. Commuters stopped as forgotten rooms and vanished faces "
            "returned in the music. Then Lio played one soft chord and saw "
            "himself leaving a brother he could no longer remember. The violin "
            "waited, bright and silent, but did he dare play the final note?"
        ),
        degraded_fixture="A musician found an instrument and played a tune.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="mars_orchard",
        prompt_class="creative",
        prompt=(
            "Write an 80-150 word story about a botanist growing the first "
            "apple on Mars. Include the exact phrase 'red dust', give the "
            "botanist a difficult choice, and make the final sentence begin "
            "with 'At dawn,'."
        ),
        rubric=(
            _check(
                "story_concepts",
                "concept_groups",
                3.0,
                groups=[
                    ["botanist"],
                    ["apple"],
                    ["mars", "martian"],
                    ["choice", "choose", "decision"],
                ],
            ),
            _check(
                "required_phrase",
                "exact_phrase",
                2.0,
                phrase="red dust",
                case_sensitive=False,
            ),
            _check(
                "required_ending",
                "regex_present",
                2.0,
                pattern=r"At dawn,[^\n]*[.!?]\s*$",
            ),
            *_common_prose_checks(min_words=80, max_words=150),
        ),
        good_fixture=(
            "Dr. Sato had coaxed one apple tree through six Martian winters, "
            "washing red dust from every leaf by hand. When its first fruit "
            "ripened, Mission Control ordered her to cut it open for sterile "
            "samples. The colony children begged to share it instead; none had "
            "ever tasted food grown beyond Earth. Keeping the apple whole "
            "could waste years of data, while obeying would turn a symbol of "
            "home into numbered slides. Sato weighed the knife against fifteen "
            "small faces pressed to the greenhouse glass. At dawn, she planted "
            "the seeds and divided the apple into fifteen careful slices."
        ),
        degraded_fixture="A scientist grew fruit on a planet and ate it.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="undersea_library",
        prompt_class="creative",
        prompt=(
            "Write an 80-150 word fable about an octopus librarian who must "
            "return a book to the surface. Include the exact phrase "
            "'salt-stained map' and finish with a one-sentence moral beginning "
            "with 'Moral:'."
        ),
        rubric=(
            _check(
                "story_concepts",
                "concept_groups",
                3.0,
                groups=[
                    ["octopus"],
                    ["librarian", "library"],
                    ["book"],
                    ["surface"],
                    ["return"],
                ],
            ),
            _check(
                "required_phrase",
                "exact_phrase",
                2.0,
                phrase="salt-stained map",
                case_sensitive=False,
            ),
            _check(
                "moral_ending",
                "regex_present",
                2.0,
                pattern=r"Moral:[^\n.!?]*[.!?]\s*$",
            ),
            *_common_prose_checks(min_words=80, max_words=150),
        ),
        good_fixture=(
            "Oona the octopus guarded the reef library and could shelve eight "
            "books at once. One morning she found a sailor's journal wrapped "
            "around a salt-stained map, stamped RETURN TO SURFACE. She feared "
            "the bright air, so she hid the book behind an atlas. Soon young "
            "fish copied her example and kept every borrowed tale. The shelves "
            "emptied, and no one could learn where the safe currents ran. "
            "Ashamed, Oona carried the journal upward in a watertight shell. A "
            "grateful sailor returned it with three new stories for the reef. "
            "Moral: Knowledge grows when even treasured things are returned."
        ),
        degraded_fixture="An octopus had a book. Moral: Books are good.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="museum_tickets",
        prompt_class="reasoning",
        prompt=(
            "A museum sold 22 tickets. Adult tickets cost $12, child tickets "
            "cost $7, and total revenue was $209. Find the number of each, show "
            "the two equations and the elimination step, and end with exactly "
            "'11 adult tickets and 11 child tickets.'"
        ),
        rubric=(
            _check(
                "problem_values",
                "concept_groups",
                2.0,
                groups=[["22"], ["12"], ["7"], ["209"]],
            ),
            _check(
                "equations",
                "concept_groups",
                2.0,
                groups=[
                    ["a + c = 22", "a+c=22"],
                    ["12a + 7c = 209", "12a+7c=209"],
                    ["5a = 55", "5a=55", "elimin"],
                ],
            ),
            _check(
                "correct_result",
                "exact_phrase",
                4.0,
                phrase="11 adult tickets and 11 child tickets",
                case_sensitive=False,
            ),
            _check(
                "required_ending",
                "ends_with",
                2.0,
                suffix="11 adult tickets and 11 child tickets.",
            ),
            *_common_prose_checks(min_words=35, max_words=140),
        ),
        good_fixture=(
            "Let a be adult tickets and c be child tickets. The count equation "
            "is a + c = 22, and the revenue equation is 12a + 7c = 209. "
            "Multiplying the first equation by 7 gives 7a + 7c = 154. "
            "Subtracting it from the revenue equation eliminates c and gives "
            "5a = 55, so a = 11 and c = 11. 11 adult tickets and 11 child "
            "tickets."
        ),
        degraded_fixture="The museum sold 12 adult tickets and 10 child tickets.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="rectangle_garden",
        prompt_class="reasoning",
        prompt=(
            "A rectangular garden's length is 3 meters more than its width, "
            "and its perimeter is 30 meters. Find the width, length, and area; "
            "show the perimeter equation and substitution; then end with "
            "exactly 'Width 6 m, length 9 m, area 54 m^2.'"
        ),
        rubric=(
            _check(
                "problem_values",
                "concept_groups",
                2.0,
                groups=[["3"], ["30"]],
            ),
            _check(
                "derivation",
                "concept_groups",
                2.0,
                groups=[
                    ["2(w + l) = 30", "2w + 2l = 30", "perimeter"],
                    ["l = w + 3", "length = width + 3"],
                    ["4w + 6 = 30", "4w=24", "4w = 24"],
                ],
            ),
            _check(
                "correct_values",
                "concept_groups",
                4.0,
                groups=[["width 6", "w = 6"], ["length 9", "l = 9"], ["54"]],
            ),
            _check(
                "required_ending",
                "ends_with",
                2.0,
                suffix="Width 6 m, length 9 m, area 54 m^2.",
            ),
            *_common_prose_checks(min_words=35, max_words=150),
        ),
        good_fixture=(
            "Let the width be w and the length be l, so l = w + 3. The "
            "perimeter equation is 2(w + l) = 30. Substituting gives "
            "2(w + w + 3) = 30, which simplifies to 4w + 6 = 30 and then "
            "4w = 24. Thus w = 6 and l = 9, and the area is 6 times 9, or 54 "
            "square meters. Width 6 m, length 9 m, area 54 m^2."
        ),
        degraded_fixture="The garden is 5 meters by 10 meters.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="discount_tax",
        prompt_class="reasoning",
        prompt=(
            "A jacket costs $80. It is discounted by 25%, then 8% sales tax is "
            "applied to the discounted price. Show the discount amount, "
            "discounted price, tax amount, and final calculation. End with "
            "exactly 'Final price: $64.80.'"
        ),
        rubric=(
            _check(
                "problem_values",
                "concept_groups",
                2.0,
                groups=[["80"], ["25%"], ["8%"]],
            ),
            _check(
                "intermediate_values",
                "concept_groups",
                3.0,
                groups=[
                    ["$20", "20 dollars", "discount amount is 20"],
                    ["$60", "60 dollars", "discounted price is 60"],
                    ["$4.80", "4.80 dollars", "tax is 4.8"],
                ],
            ),
            _check(
                "correct_result",
                "exact_phrase",
                4.0,
                phrase="Final price: $64.80",
                case_sensitive=False,
            ),
            _check(
                "required_ending",
                "ends_with",
                2.0,
                suffix="Final price: $64.80.",
            ),
            *_common_prose_checks(min_words=30, max_words=130),
        ),
        good_fixture=(
            "The discount amount is 25% of $80, so 0.25 times 80 equals $20. "
            "Subtracting gives a discounted price of $60. The sales tax is 8% "
            "of $60, so 0.08 times 60 equals $4.80. Adding the tax to the "
            "discounted price gives $60 + $4.80 = $64.80. Final price: $64.80."
        ),
        degraded_fixture="Add 25% and 8%, so the jacket costs $106.40.",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="dedupe_order_code",
        prompt_class="code",
        prompt=(
            "Write only a Python function `def dedupe_preserve_order(items: "
            "list) -> list:` that returns the first occurrence of each item in "
            "the original order. Do not use imports or markdown fences."
        ),
        rubric=(
            _check(
                "python_contract",
                "python_ast_contract_v2",
                8.0,
                function_name="dedupe_preserve_order",
                arguments=["items"],
                return_annotation_contains="list",
                required_source_patterns=[
                    r"for\s+\w+\s+in\s+items",
                    r"if\s+\w+\s+not\s+in",
                    r"\.append\s*\(",
                ],
                require_return=True,
            ),
            _check("no_markdown_fence", "no_markdown_fence", 2.0),
        ),
        good_fixture=(
            "def dedupe_preserve_order(items: list) -> list:\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in result:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        degraded_fixture="def dedupe(items):\n    return list(set(items))\n",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="clamp_code",
        prompt_class="code",
        prompt=(
            "Write only a Python function `def clamp(value: float, minimum: "
            "float, maximum: float) -> float:` that raises ValueError when "
            "minimum is greater than maximum and otherwise bounds value to the "
            "inclusive range. Do not use imports or markdown fences."
        ),
        rubric=(
            _check(
                "python_contract",
                "python_ast_contract_v2",
                8.0,
                function_name="clamp",
                arguments=["value", "minimum", "maximum"],
                return_annotation_contains="float",
                required_source_patterns=[
                    r"minimum\s*>\s*maximum",
                    r"raise\s+ValueError",
                    r"max\s*\(\s*minimum",
                    r"min\s*\(\s*maximum",
                ],
                require_return=True,
                require_raise=True,
            ),
            _check("no_markdown_fence", "no_markdown_fence", 2.0),
        ),
        good_fixture=(
            "def clamp(value: float, minimum: float, maximum: float) -> float:\n"
            "    if minimum > maximum:\n"
            "        raise ValueError('minimum exceeds maximum')\n"
            "    return max(minimum, min(maximum, value))\n"
        ),
        degraded_fixture="def clamp(value):\n    return value\n",
        evaluator_id=EVALUATOR_V2_ID,
    ),
    BasinPromptSpec(
        key="word_frequencies_code",
        prompt_class="code",
        prompt=(
            "Write only a Python function `def word_frequencies(words: "
            "list[str]) -> dict[str, int]:` that counts words "
            "case-insensitively without imports. Do not use markdown fences."
        ),
        rubric=(
            _check(
                "python_contract",
                "python_ast_contract_v2",
                8.0,
                function_name="word_frequencies",
                arguments=["words"],
                return_annotation_contains="dict",
                required_source_patterns=[
                    r"for\s+\w+\s+in\s+words",
                    r"\.casefold\s*\(",
                    r"\.get\s*\(",
                ],
                require_return=True,
            ),
            _check("no_markdown_fence", "no_markdown_fence", 2.0),
        ),
        good_fixture=(
            "def word_frequencies(words: list[str]) -> dict[str, int]:\n"
            "    counts = {}\n"
            "    for word in words:\n"
            "        key = word.casefold()\n"
            "        counts[key] = counts.get(key, 0) + 1\n"
            "    return counts\n"
        ),
        degraded_fixture="def count(words):\n    return len(words)\n",
        evaluator_id=EVALUATOR_V2_ID,
    ),
)

ALL_PROMPT_SPECS = (*PROMPT_SPECS, *M3R2_PROMPT_SPECS)
_PROMPT_TABLE = {spec.key: spec for spec in ALL_PROMPT_SPECS}


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


def _extract_python_function(
    text: str,
    *,
    function_name: str,
) -> tuple[str | None, ast.Module | None, int | None]:
    lines = str(text).expandtabs().splitlines()
    declaration = re.compile(
        rf"^\s*(?:async\s+)?def\s+{re.escape(function_name)}\s*\("
    )
    for start, line in enumerate(lines):
        if declaration.search(line) is None:
            continue
        for end in range(len(lines), start, -1):
            candidate = textwrap.dedent(
                "\n".join(lines[start:end])
            ).strip()
            if not candidate:
                continue
            try:
                tree = ast.parse(candidate)
            except (SyntaxError, ValueError):
                continue
            found = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
                for node in tree.body
            )
            if found:
                return candidate, tree, start + 1
        return None, None, start + 1
    return None, None, None


def _python_contract_v2_score(
    text: str,
    check: Mapping[str, Any],
) -> tuple[float, Any]:
    function_name = str(check["function_name"])
    source, tree, start_line = _extract_python_function(
        text,
        function_name=function_name,
    )
    if source is None or tree is None:
        return 0.0, {
            "parsed": False,
            "extracted": start_line is not None,
            "source_start_line": start_line,
            "criteria": [],
        }

    criteria: list[tuple[str, bool]] = [("parsed", True)]
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
            {
                "parsed": True,
                "extracted": True,
                "source_start_line": start_line,
                "criteria": criteria,
            },
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
                re.search(
                    str(pattern),
                    source,
                    flags=re.IGNORECASE,
                )
                is not None,
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
        {
            "parsed": True,
            "extracted": True,
            "source_start_line": start_line,
            "source_character_count": len(source),
            "criteria": criteria,
        },
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
    if kind == "numbered_steps_max":
        count = len(
            re.findall(
                r"(?m)^\s*\d+[.)]\s+",
                text,
            )
        )
        maximum = int(check["maximum"])
        return float(count <= maximum), {"numbered_steps": count}
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
    if kind == "python_ast_contract_v2":
        return _python_contract_v2_score(text, check)
    raise ValueError(f"Unknown deterministic rubric check kind: {kind!r}")


def score_output(spec: BasinPromptSpec, text: str) -> dict[str, Any]:
    """Score one output without receiving or inferring its branch identity."""
    if spec.evaluator_id not in SUPPORTED_EVALUATOR_IDS:
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
        "protocol": (
            "branch-blind-deterministic-paired-scoring-v2"
            if spec.evaluator_id == EVALUATOR_V2_ID
            else "branch-blind-deterministic-paired-scoring-v1"
        ),
        "branch_identity_hidden_from_scorer": True,
        "tie_tolerance": tolerance,
        "clean": clean,
        "perturbed": perturbed,
        "score_delta": delta,
        "outcome": outcome,
    }


def validate_prompt_specs(
    prompt_specs: Iterable[BasinPromptSpec] = PROMPT_SPECS,
) -> dict[str, Any]:
    selected = tuple(prompt_specs)
    prompts: dict[str, Any] = {}
    for spec in selected:
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
        "evaluator_id": (
            selected[0].evaluator_id
            if selected
            and len({spec.evaluator_id for spec in selected}) == 1
            else None
        ),
        "evaluator_ids": sorted(
            {spec.evaluator_id for spec in selected}
        ),
        "prompt_count": len(selected),
        "all_passed": all(row["passed"] for row in prompts.values()),
        "prompts": prompts,
    }
