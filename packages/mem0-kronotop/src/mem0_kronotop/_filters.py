# Copyright (c) 2026 Burak Sezer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mem0 filters, as BQL.

A mem0 filter names a field of the payload. The store keeps the payload under
``payload``, so every field gets that prefix.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = ["to_bql"]

#: mem0 spells a logical operator two ways. ``Memory._process_metadata_filters``
#: writes the ``$`` form. A filter that skips it keeps the upper case one.
_LOGICAL = {
    "AND": "$and",
    "$and": "$and",
    "OR": "$or",
    "$or": "$or",
    "NOT": "$nor",
    "$not": "$nor",
}

_COMPARISON = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
}


def _condition(field: str, condition: Any) -> Any:
    """What one field has to match."""
    if condition == "*":
        return {"$exists": True}
    if not isinstance(condition, Mapping):
        return condition
    bql: dict[str, Any] = {}
    for operator, value in condition.items():
        if operator in _COMPARISON:
            bql[_COMPARISON[operator]] = value
        elif operator in ("contains", "icontains"):
            # Escaped, so the text is matched literally.
            bql["$regex"] = re.escape(value)
            if operator == "icontains":
                bql["$options"] = "i"
        else:
            raise ValueError(f"filter operator {operator!r} on {field!r} is not known")
    return bql


def to_bql(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """The BQL query that matches what ``filters`` matches.

    No filter gives ``{}``, which matches every document.
    """
    bql: dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if key not in _LOGICAL:
            bql[f"payload.{key}"] = _condition(key, value)
            continue
        operator = _LOGICAL[key]
        # The same conditions can arrive under both spellings. The first wins.
        if operator not in bql:
            bql[operator] = [to_bql(condition) for condition in value]
    return bql
