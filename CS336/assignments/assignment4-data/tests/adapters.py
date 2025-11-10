from __future__ import annotations

import os
from typing import Any
import cs336_data



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return cs336_data.run_extract_text_from_html_bytes_implementation(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return cs336_data.run_identify_language_implementation(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return cs336_data.run_mask_emails_implementation(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return cs336_data.run_mask_phone_numbers_implementation(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return cs336_data.run_mask_ip_addresses_implementation(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return cs336_data.run_classify_nsfw_implementation(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return cs336_data.run_classify_toxic_speech_implementation(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return cs336_data.run_gopher_quality_filter_implementation(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return cs336_data.run_exact_line_deduplication_implementation(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return cs336_data.run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, output_directory)
