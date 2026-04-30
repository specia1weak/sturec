from typing import Dict

import polars as pl
import torch


NULL_FALLBACK = "NULL_FALLBACK"


def explode_expr(field_name: str, is_string_format: bool = True, separator: str = ",") -> pl.Expr:
    expr = pl.col(field_name).drop_nulls()
    if is_string_format:
        exploded = (
            expr.cast(pl.Utf8)
            .str.split(separator)
            .explode()
            .cast(pl.Utf8)
            .str.strip_chars()
        )
    else:
        exploded = expr.explode().cast(pl.Utf8)
    valid_mask = exploded.is_not_null() & (exploded != "")
    return exploded.filter(valid_mask)


def clear_seq_expr(
    field_name: str,
    is_string_format: bool,
    separator: str,
    fill_empty_with_fallback: bool = False,
) -> pl.Expr:
    if is_string_format:
        expr = (
            pl.col(field_name)
            .cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        )
    else:
        expr = (
            pl.col(field_name)
            .cast(pl.List(pl.Utf8))
            .fill_null([])
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
        )

    if not fill_empty_with_fallback:
        return expr

    return (
        pl.when(expr.list.len() > 0)
        .then(expr)
        .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
    )


def seq_length_expr(
    field_name: str,
    is_string_format: bool,
    separator: str,
    alias: str,
) -> pl.Expr:
    clean_expr = clear_seq_expr(
        field_name=field_name,
        is_string_format=is_string_format,
        separator=separator,
        fill_empty_with_fallback=False,
    )
    return clean_expr.list.len().cast(pl.UInt32).alias(alias)


def map_list_to_indices(expr: pl.Expr, vocab: Dict[str, int], oov_idx: int) -> pl.Expr:
    keys = pl.Series(list(vocab.keys()), dtype=pl.Utf8)
    vals = pl.Series(list(vocab.values()), dtype=pl.UInt32)
    default_expr = pl.lit(oov_idx, dtype=pl.UInt32) if oov_idx >= 0 else pl.lit(None, dtype=pl.UInt32)
    return expr.list.eval(
        pl.element()
        .replace_strict(old=keys, new=vals, default=default_expr)
        .cast(pl.UInt32)
    )


def mean_pooling(emb: torch.Tensor, idx_tensor: torch.Tensor, padding_zero: bool) -> torch.Tensor:
    if padding_zero:
        mask_sum = (idx_tensor != 0).sum(dim=-1, keepdim=True).clamp(min=1)
    else:
        mask_shape = idx_tensor.shape[:-1] + (1,)
        mask_sum = torch.full(mask_shape, idx_tensor.shape[-1], dtype=emb.dtype, device=idx_tensor.device)
        return emb / mask_sum
    return emb / mask_sum.to(dtype=emb.dtype)
