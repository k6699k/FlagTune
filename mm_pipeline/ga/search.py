#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Genetic config generation helpers for mm top-k expansion.

This module is intentionally runtime-agnostic: it only flattens, deduplicates,
crosses over, mutates, and generates config dictionaries. The caller owns
benchmarking and may feed measured latency back through ``ga_latency_ms``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


CONFIG_FIELDS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_M",
    "SPLIT_K",
    "num_warps",
    "num_ctas",
    "num_stages",
]

LEGAL_VALUE_SPACE_BY_KERNEL = {
    # Must match src/flag_gems/runtime/backend/_nvidia/hopper/mm_hopper_expand.yaml.
    # 5 * 4 * 4 * 7 * 3 * 2 = 3360 legal mm_general_tma configs.
    "mm_general_tma": {
        "BLOCK_M": [16, 32, 64, 128, 256],
        "BLOCK_N": [16, 32, 64, 128],
        "BLOCK_K": [32, 64, 128, 256],
        "GROUP_M": [1, 2, 4, 8, 16, 32, 64],
        "num_stages": [2, 3, 4],
        "num_warps": [4, 8],
    },
    "mm_general": {
        "BLOCK_M": [16, 32, 64, 128, 256],
        "BLOCK_N": [16, 32, 64, 128],
        "BLOCK_K": [32, 64, 128, 256],
        "GROUP_M": [1, 2, 4, 8, 16, 32, 64],
        "num_stages": [2, 3, 4],
        "num_warps": [4, 8],
    },
    # Must match src/flag_gems/runtime/backend/_nvidia/hopper/mm_hopper_expand.yaml.
    "mm_splitk": {
        "BLOCK_M": [8, 16],
        "BLOCK_N": [16, 32],
        "BLOCK_K": [64, 128, 256],
        "SPLIT_K": [4, 8, 16, 32],
        "num_stages": [2, 3, 4, 5, 6, 8],
        "num_warps": [2, 4],
    },
    # 3 * 2 * 7 * 4 = 168 legal gemv configs.
    "gemv": {
        "BLOCK_M": [8, 16, 32],
        "BLOCK_K": [128, 256],
        "num_stages": [2, 3, 4, 5, 6, 7, 8],
        "num_warps": [1, 2, 4, 8],
    },
}


@dataclass(frozen=True)
class GAOptions:
    generations: int
    population_size: int
    elite_size: int
    offspring_per_generation: int
    mutation_rate: float
    random_rate: float
    max_evaluations: int = 0


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def normalize_kernel_kind(kernel_kind: str) -> str:
    text = str(kernel_kind).strip().lower()
    if text in ("gemv", "mv", "matrix_vector"):
        return "gemv"
    if text in ("mm_splitk", "splitk", "split_k"):
        return "mm_splitk"
    if text in ("mm_general", "general", "non_tma", "mm_general_non_tma"):
        return "mm_general"
    return "mm_general_tma"


def legal_value_space(kernel_kind: str) -> Dict[str, List[int]]:
    kind = normalize_kernel_kind(kernel_kind)
    return {field: list(values) for field, values in LEGAL_VALUE_SPACE_BY_KERNEL[kind].items()}


def active_config_fields(kernel_kind: str) -> Tuple[str, ...]:
    return tuple(legal_value_space(kernel_kind).keys())


def flatten_config(config: Dict[str, Any]) -> Dict[str, Optional[int]]:
    meta = config.get("META", {}) if isinstance(config, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    group_m = config.get("GROUP_M", meta.get("GROUP_M", 8))
    return {
        "BLOCK_M": parse_int(config.get("BLOCK_M", meta.get("BLOCK_M"))),
        "BLOCK_N": parse_int(config.get("BLOCK_N", meta.get("BLOCK_N"))),
        "BLOCK_K": parse_int(config.get("BLOCK_K", meta.get("BLOCK_K"))),
        "GROUP_M": parse_int(group_m) or 8,
        "SPLIT_K": parse_int(config.get("SPLIT_K", meta.get("SPLIT_K"))),
        "num_warps": parse_int(config.get("num_warps")),
        "num_ctas": parse_int(config.get("num_ctas")),
        "num_stages": parse_int(config.get("num_stages")),
    }


def config_key(flat: Dict[str, Any], kernel_kind: str) -> Tuple[Tuple[str, str], ...]:
    pairs = []
    for field in active_config_fields(kernel_kind):
        value = flat.get(field)
        if value is None:
            continue
        pairs.append((field, str(int(value))))
    return tuple(pairs)


def entry_key(entry: Dict[str, Any], kernel_kind: str) -> Tuple[Tuple[str, str], ...]:
    return config_key(flatten_config(entry["config"]), kernel_kind)


def config_from_flat(flat: Dict[str, Any], kernel_kind: str) -> Dict[str, Any]:
    meta = {
        "BLOCK_M": int(flat["BLOCK_M"]),
        "BLOCK_K": int(flat["BLOCK_K"]),
    }
    if kernel_kind != "gemv":
        meta["BLOCK_N"] = int(flat["BLOCK_N"])
        if kernel_kind == "mm_splitk":
            meta["SPLIT_K"] = int(flat["SPLIT_K"])
        else:
            meta["GROUP_M"] = int(flat["GROUP_M"])

    config: Dict[str, Any] = {
        "META": meta,
        "num_warps": int(flat["num_warps"]),
        "num_stages": int(flat["num_stages"]),
    }
    return config


def clone_entry(
    base: Dict[str, Any],
    config: Dict[str, Any],
    generation: int,
    source: str,
    candidate_rank: Optional[int] = None,
) -> Dict[str, Any]:
    entry = dict(base)
    entry["config"] = config
    entry["ga_generation"] = generation
    entry["ga_source"] = source
    if candidate_rank is not None:
        entry["candidate_rank"] = candidate_rank
    elif generation > 0:
        entry.pop("candidate_rank", None)
    return entry


def initial_population(
    entries: Sequence[Dict[str, Any]], kernel_kind: str
) -> List[Dict[str, Any]]:
    population = []
    seen = set()
    for entry in sorted(
        entries,
        key=lambda item: (
            parse_int(item.get("candidate_rank")) is None,
            parse_int(item.get("candidate_rank")) or 0,
        ),
    ):
        cloned = clone_entry(
            entry,
            entry["config"],
            generation=0,
            source="topk",
            candidate_rank=parse_int(entry.get("candidate_rank")),
        )
        key = entry_key(cloned, kernel_kind)
        if key in seen:
            continue
        seen.add(key)
        population.append(cloned)
    return population


def value_space(entries: Sequence[Dict[str, Any]], kernel_kind: str) -> Dict[str, List[int]]:
    return legal_value_space(kernel_kind)


def validate_flat(flat: Dict[str, Any], kernel_kind: str) -> bool:
    values = legal_value_space(kernel_kind)
    for field, choices in values.items():
        value = flat.get(field)
        if value is None or int(value) <= 0:
            return False
        if int(value) not in choices:
            return False
    return True


def crossover(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    kernel_kind: str,
    rng: random.Random,
) -> Dict[str, Any]:
    flat_a = flatten_config(parent_a["config"])
    flat_b = flatten_config(parent_b["config"])
    child = {}
    for field in active_config_fields(kernel_kind):
        value = flat_a.get(field) if rng.random() < 0.5 else flat_b.get(field)
        if value is not None:
            child[field] = int(value)
    return child


def mutate(
    flat: Dict[str, Any],
    values: Dict[str, List[int]],
    kernel_kind: str,
    mutation_rate: float,
    rng: random.Random,
) -> Dict[str, Any]:
    out = dict(flat)
    for field, choices in values.items():
        if not choices:
            continue
        if field not in out or rng.random() < mutation_rate:
            out[field] = int(rng.choice(choices))
    return out


def random_flat(
    values: Dict[str, List[int]], kernel_kind: str, rng: random.Random
) -> Dict[str, Any]:
    flat = {}
    for field, choices in values.items():
        if choices:
            flat[field] = int(rng.choice(choices))
    return flat


def parent_pool(
    known_entries: Sequence[Dict[str, Any]],
    kernel_kind: str,
    options: GAOptions,
) -> List[Dict[str, Any]]:
    seen = set()
    parents = []

    def add(entry: Dict[str, Any]) -> None:
        key = entry_key(entry, kernel_kind)
        if key in seen:
            return
        seen.add(key)
        parents.append(entry)

    ranked_entries = sorted(
        known_entries,
        key=lambda item: (
            item.get("ga_latency_ms") is None,
            float(item.get("ga_latency_ms") or 0.0),
            parse_int(item.get("candidate_rank")) is None,
            parse_int(item.get("candidate_rank")) or 0,
        ),
    )
    for entry in ranked_entries[: max(1, options.elite_size)]:
        add(entry)
    for entry in ranked_entries:
        if len(parents) >= max(1, options.population_size):
            break
        add(entry)
    return parents or list(known_entries[:1])


def next_generation(
    base_entry: Dict[str, Any],
    known_entries: Sequence[Dict[str, Any]],
    known_keys: set[Tuple[Tuple[str, str], ...]],
    kernel_kind: str,
    generation: int,
    target_count: int,
    options: GAOptions,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    parents = parent_pool(known_entries, kernel_kind, options)
    values = value_space(known_entries, kernel_kind)
    offspring: List[Dict[str, Any]] = []
    batch_keys = set()
    max_attempts = max(100, target_count * 50)
    attempts = 0
    while len(offspring) < target_count and attempts < max_attempts:
        attempts += 1
        if rng.random() < options.random_rate or len(parents) == 1:
            child_flat = random_flat(values, kernel_kind, rng)
            source = "random"
        else:
            parent_a, parent_b = rng.sample(parents, 2)
            child_flat = crossover(parent_a, parent_b, kernel_kind, rng)
            source = "crossover"
        child_flat = mutate(
            child_flat, values, kernel_kind, options.mutation_rate, rng
        )
        if not validate_flat(child_flat, kernel_kind):
            continue
        key = config_key(child_flat, kernel_kind)
        if key in known_keys or key in batch_keys:
            continue
        batch_keys.add(key)
        offspring.append(
            clone_entry(
                base_entry,
                config_from_flat(child_flat, kernel_kind),
                generation=generation,
                source=source,
            )
        )
    return offspring


def generate_configs(
    entries: Sequence[Dict[str, Any]],
    kernel_kind: str,
    options: GAOptions,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    initial_entries = initial_population(entries, kernel_kind)
    if (
        not initial_entries
        or options.generations <= 0
        or options.offspring_per_generation <= 0
    ):
        return []

    generated_limit = options.offspring_per_generation
    if options.max_evaluations:
        generated_limit = min(
            generated_limit,
            max(0, options.max_evaluations - len(initial_entries)),
        )
    if generated_limit <= 0:
        return []

    known_entries = list(initial_entries)
    known_keys = {entry_key(entry, kernel_kind) for entry in known_entries}
    generated_history: List[Dict[str, Any]] = []
    base_entry = entries[0]

    for generation in range(1, options.generations + 1):
        offspring = next_generation(
            base_entry,
            known_entries,
            known_keys,
            kernel_kind,
            generation,
            options.offspring_per_generation,
            options,
            rng,
        )
        if not offspring:
            continue
        for entry in offspring:
            known_keys.add(entry_key(entry, kernel_kind))
        known_entries.extend(offspring)
        generated_history.extend(offspring)

    return generated_history[-generated_limit:]
