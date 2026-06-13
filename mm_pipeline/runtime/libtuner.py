#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional

from FlagTune.mm_pipeline.ga import libtuner_runtime as mm_ga_libtuner_runtime
from FlagTune.mm_pipeline.runtime.common import normalize_kernel_kind, parse_float, parse_int
from FlagTune.mm_pipeline.runtime.configs import configs_match, find_matching_config_entry, flatten_config_record, triton_config_record


KERNEL_ATTR_BY_KIND = {
    "gemv": "gemv_kernel",
    "mm_general_tma": "mm_kernel_general_host_tma",
    "mm_general": "mm_kernel_general",
    "mm_splitk": "mm_kernel_splitk",
}


def unwrap_tuner(kernel: Any) -> Any:
    missing = object()
    fn = kernel
    visited = set()
    while fn is not None and id(fn) not in visited:
        visited.add(id(fn))
        if (
            safe_getattr(fn, "configs", missing) is not missing
            and safe_getattr(fn, "policy", missing) is not missing
            and safe_getattr(fn, "prune_configs", missing) is not missing
        ):
            return fn
        fn = safe_getattr(fn, "fn", None)
    return None


def is_libentry_kernel(kernel: Any) -> bool:
    missing = object()
    return (
        safe_getattr(kernel, "kernel_cache", missing) is not missing
        and safe_getattr(kernel, "key", missing) is not missing
        and safe_getattr(kernel, "fn", missing) is not missing
    )


def kernel_attr_for_kind(kernel_kind: str) -> str:
    kind = normalize_kernel_kind(kernel_kind) or kernel_kind
    try:
        return KERNEL_ATTR_BY_KIND[kind]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported kernel kind: {kernel_kind}") from exc


def find_kernel_libentry_with_module(
    kernel_kind: str, preferred_module: Any
) -> tuple[Any, Any, Optional[str], Any]:
    kernel_attr = kernel_attr_for_kind(kernel_kind)
    candidates: List[tuple[Any, Any, str, Any]] = []
    for module_name, module in iter_loaded_mm_modules(preferred_module):
        kernel = safe_getattr(module, kernel_attr, None)
        if not is_libentry_kernel(kernel):
            continue
        tuner = unwrap_tuner(kernel)
        if tuner is None:
            continue
        candidates.append((kernel, tuner, module_name, module))
        if safe_getattr(tuner, "best_config", None) is not None:
            return kernel, tuner, module_name, module

    if candidates:
        return candidates[0]
    return None, None, None, None


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def runtime_config_dict(config_entry: Dict[str, Any]) -> Dict[str, Any]:
    config = config_entry.get("config", config_entry)
    if not isinstance(config, dict):
        raise ValueError(f"Missing config object in runtime config: {config_entry!r}")
    return config


def runtime_config_int(
    config: Dict[str, Any],
    meta: Dict[str, Any],
    name: str,
    *,
    required: bool = False,
) -> Optional[int]:
    value = config.get(name, meta.get(name))
    if value is None:
        if required:
            raise ValueError(f"runtime config requires {name}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} in runtime config: {value!r}") from exc


def make_triton_config(
    meta: Dict[str, Any],
    *,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    num_ctas: Optional[int] = None,
    pre_hook: Any = None,
) -> Any:
    import triton

    kwargs: Dict[str, Any] = {}
    if num_warps is not None:
        kwargs["num_warps"] = num_warps
    if num_stages is not None:
        kwargs["num_stages"] = num_stages
    if pre_hook is not None:
        kwargs["pre_hook"] = pre_hook
    if num_ctas is not None:
        kwargs["num_ctas"] = num_ctas
    try:
        return triton.Config(meta, **kwargs)
    except TypeError:
        if "num_ctas" not in kwargs:
            raise
        kwargs.pop("num_ctas", None)
        return triton.Config(meta, **kwargs)


def make_mm_tma_runtime_configs(
    config_entries: List[Dict[str, Any]],
    *,
    pre_hook: Any,
) -> List[Any]:
    if pre_hook is None:
        raise RuntimeError("mm_general_tma LibTuner does not expose a FlagTune pre-hook")

    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        block_m = runtime_config_int(config, meta, "BLOCK_M", required=True)
        block_n = runtime_config_int(config, meta, "BLOCK_N", required=True)
        block_k = runtime_config_int(config, meta, "BLOCK_K", required=True)
        group_m = runtime_config_int(config, meta, "GROUP_M") or 8
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                },
                num_warps=runtime_config_int(config, meta, "num_warps", required=True),
                num_stages=runtime_config_int(
                    config, meta, "num_stages", required=True
                ),
                num_ctas=runtime_config_int(config, meta, "num_ctas"),
                pre_hook=pre_hook,
            )
        )
    return configs


def make_mm_general_runtime_configs(config_entries: List[Dict[str, Any]]) -> List[Any]:
    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": runtime_config_int(config, meta, "BLOCK_M", required=True),
                    "BLOCK_N": runtime_config_int(config, meta, "BLOCK_N", required=True),
                    "BLOCK_K": runtime_config_int(config, meta, "BLOCK_K", required=True),
                    "GROUP_M": runtime_config_int(config, meta, "GROUP_M") or 8,
                },
                num_warps=runtime_config_int(config, meta, "num_warps", required=True),
                num_stages=runtime_config_int(
                    config, meta, "num_stages", required=True
                ),
                num_ctas=runtime_config_int(config, meta, "num_ctas"),
            )
        )
    return configs


def make_mm_splitk_runtime_configs(config_entries: List[Dict[str, Any]]) -> List[Any]:
    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": runtime_config_int(config, meta, "BLOCK_M", required=True),
                    "BLOCK_N": runtime_config_int(config, meta, "BLOCK_N", required=True),
                    "BLOCK_K": runtime_config_int(config, meta, "BLOCK_K", required=True),
                    "SPLIT_K": runtime_config_int(config, meta, "SPLIT_K") or 4,
                },
                num_warps=runtime_config_int(config, meta, "num_warps", required=True),
                num_stages=runtime_config_int(
                    config, meta, "num_stages", required=True
                ),
            )
        )
    return configs


def make_gemv_runtime_configs(config_entries: List[Dict[str, Any]]) -> List[Any]:
    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": runtime_config_int(config, meta, "BLOCK_M", required=True),
                    "BLOCK_K": runtime_config_int(config, meta, "BLOCK_K", required=True),
                },
                num_warps=runtime_config_int(config, meta, "num_warps"),
                num_stages=runtime_config_int(config, meta, "num_stages"),
            )
        )
    return configs


def iter_loaded_mm_modules(preferred_module: Any) -> Iterable[tuple[str, Any]]:
    seen = set()
    if preferred_module is not None:
        seen.add(id(preferred_module))
        yield safe_getattr(preferred_module, "__name__", "<preferred>"), preferred_module

    for name, module in list(sys.modules.items()):
        if not isinstance(name, str) or not name.endswith(".ops.mm"):
            continue
        if id(module) in seen:
            continue
        if (
            safe_getattr(module, "gemv_kernel", None) is None
            and safe_getattr(module, "mm_kernel_general_host_tma", None) is None
            and safe_getattr(module, "mm_kernel_general", None) is None
            and safe_getattr(module, "mm_kernel_splitk", None) is None
        ):
            continue
        seen.add(id(module))
        yield name, module


def find_kernel_tuner_with_module(
    kernel_kind: str, preferred_module: Any
) -> tuple[Any, Optional[str], Any]:
    kernel_attr = kernel_attr_for_kind(kernel_kind)
    candidates: List[tuple[Any, str, Any]] = []
    for module_name, module in iter_loaded_mm_modules(preferred_module):
        kernel = safe_getattr(module, kernel_attr, None)
        tuner = unwrap_tuner(kernel)
        if tuner is None:
            continue
        candidates.append((tuner, module_name, module))
        if safe_getattr(tuner, "best_config", None) is not None:
            return tuner, module_name, module

    if candidates:
        return candidates[0]
    return None, None, None


def make_runtime_configs(
    kernel_kind: str,
    payload: Dict[str, Any],
    *,
    pre_hook: Any = None,
) -> List[Any]:
    config_entries = payload.get("configs")
    if not isinstance(config_entries, list) or not config_entries:
        raise RuntimeError("Worker payload does not contain configs")

    if kernel_kind == "gemv":
        return make_gemv_runtime_configs(config_entries)
    if kernel_kind == "mm_general_tma":
        return make_mm_tma_runtime_configs(config_entries, pre_hook=pre_hook)
    if kernel_kind == "mm_general":
        return make_mm_general_runtime_configs(config_entries)
    if kernel_kind == "mm_splitk":
        return make_mm_splitk_runtime_configs(config_entries)
    raise RuntimeError(f"Unsupported kernel kind for runtime configs: {kernel_kind}")


def libtuner_pre_hook(tuner: Any) -> Any:
    pre_hook = safe_getattr(tuner, "_flagtune_pre_hook", None)
    if pre_hook is not None:
        return pre_hook
    for config in safe_getattr(tuner, "configs", []) or []:
        pre_hook = safe_getattr(config, "pre_hook", None)
        if pre_hook is not None:
            return pre_hook
    return None


def install_runtime_configs(kernel_kind: str, payload: Dict[str, Any]) -> tuple[Any, Optional[str]]:
    tuner, tuner_source, module = find_kernel_tuner_with_module(
        kernel_kind, None)
    if tuner is None or module is None:
        raise RuntimeError(
            f"Could not find LibTuner for kernel kind: {kernel_kind}")
    pre_hook = libtuner_pre_hook(tuner)
    tuner._flagtune_runtime_configs = make_runtime_configs(
        kernel_kind, payload, pre_hook=pre_hook)
    return tuner, tuner_source


def ga_libtuner_runtime_helpers() -> mm_ga_libtuner_runtime.LibTunerRuntimeHelpers:
    return mm_ga_libtuner_runtime.LibTunerRuntimeHelpers(
        install_runtime_configs=install_runtime_configs,
        triton_config_record=triton_config_record,
        find_matching_config_entry=find_matching_config_entry,
        flatten_config_record=flatten_config_record,
        configs_match=configs_match,
        parse_float=parse_float,
        parse_int=parse_int,
        safe_getattr=safe_getattr,
    )
