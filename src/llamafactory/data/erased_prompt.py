# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Erased prompt presets of the CIG-SFT loss.

The erased prompt replaces the whole user message while the response is kept, so the frozen reference model can be
asked for the probability of the same response with and without the instance content. The presets keep the task
instruction and drop only the instance, which is what makes the two views comparable across a dataset.
"""

from typing import Optional


#: The column that `align_dataset` carries the erased prompt of every example in.
ERASED_PROMPT_COLUMN = "_erased_prompt"

ERASED_PROMPT_PRESETS: dict[str, str] = {
    "math": r"[PROBLEM_REMOVED]" "\n" r"Let's think step by step and output the final answer within \boxed{}.",
    "code": "[TASK_REMOVED]",
    "medical": "Question:\n[QUESTION_REMOVED]\n\nOptions:\n[OPTIONS_REMOVED]\n\nAnswer:",
    "default": "[CONTEXT_REMOVED]",
}


def resolve_erased_prompt(value: Optional[str]) -> Optional[str]:
    r"""Resolve a preset name into its placeholder text, or pass a literal prompt through.

    Non-string and blank values count as unset, so that a dataset column with a missing entry falls back to the
    dataset level prompt instead of erasing the instance with an empty message.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    return ERASED_PROMPT_PRESETS.get(value.strip().lower(), value)
