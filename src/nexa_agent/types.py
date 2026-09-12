"""Shared low-level types for NEXA's portable agent layer."""

from __future__ import annotations

from typing import Literal

# Pydantic needs PEP 695 named recursive aliases for JSON-like values.
# 一个 JSON 值可以是字符串、数字、布尔值或 None。
type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]

# 流式增量片段的类别：正文（text）或思考（reasoning）。
# Provider 层和 Agent 层共用同一个定义，避免两处各写一遍。
type DeltaKind = Literal["text", "reasoning"]
