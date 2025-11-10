from __future__ import annotations
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from typing import Any
import re
import fasttext
import os
from xopen import xopen
from pathlib import Path
from typing import List, Dict, Any, Set
import hashlib
import os
import unicodedata
import random
seed=42
ft = fasttext.load_model('./models/lid.176.bin')
ft_nsfw = fasttext.load_model('./models/jigsaw_fasttext_bigrams_nsfw_final.bin')
ft_toxic = fasttext.load_model('./models/jigsaw_fasttext_bigrams_hatespeech_final.bin')

email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
phone_pattern = r'''
    (
        (?:(?:\+?1[\s\-\.]?)?)        # optional country code (+1, 1-, 1.)
        (?:\(?\d{3}\)?[\s\-\.]?)?     # optional area code with or without parentheses
        \d{3}[\s\-\.]?\d{4}           # main 7 digits
    )
    |
    (\b\d{10}\b)                      # OR plain 10 digits (no separators)
'''

def run_extract_text_from_html_bytes_implementation(html_bytes: bytes):
    detected = detect_encoding(html_bytes)
    encode_way = detected.encoding if hasattr(detected, "encoding") else detected or "utf-8"
    
    try:
        # First try the detected encoding
        return extract_plain_text(html_bytes.decode(encoding=encode_way))
    except UnicodeDecodeError:
        # Then try more permissive encodings
        for fallback in ["latin-1", "windows-1252"]:
            try:
                return extract_plain_text(html_bytes.decode(encoding=fallback))
            except UnicodeDecodeError:
                continue
        # Final fallback: ignore undecodable bytes
        return extract_plain_text(html_bytes.decode(encoding="utf-8", errors="ignore"))

def run_identify_language_implementation(text:str) -> tuple[Any, float]: 
    text = text.replace('\n', " ")
    label, score = ft.predict(text)
    if 'zh' in label[0]:
        label = 'zh'
    elif 'en' in label[0]:
        label = 'en'
    score = score[0]
    return label, score


def run_mask_emails_implementation(text:str) -> tuple[str, int]:
    result_str, counts = re.subn(email_pattern,"|||EMAIL_ADDRESS|||", text) 
    return  result_str, counts

def run_mask_phone_numbers_implementation(text: str) -> tuple[str, int]:
    """
    掩码文本中的电话号码
    """
    # 简化的电话号码模式
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    # 使用subn进行替换
    result_str, counts = re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)
    return result_str, counts

def run_mask_ip_addresses_implementation(text: str) -> tuple[str, int]:
    """
    掩码文本中的 IPv4 地址
    
    IPv4 地址格式：四个 0-255 的数字用点分隔
    例如：192.168.1.1, 10.0.0.1, 255.255.255.0
    """
    
    # IPv4 地址正则表达式
    ip_pattern = r'''
        \b                          # 单词边界
        (?:                         # 非捕获分组
            (?:25[0-5]|             # 250-255
            2[0-4][0-9]|           # 200-249
            [01]?[0-9][0-9]?)      # 0-199
            \.                      # 点分隔符
        ){3}                        # 重复3次（前三个数字）
        (?:25[0-5]|                 # 最后一个数字
        2[0-4][0-9]|
        [01]?[0-9][0-9]?)
        \b                          # 单词边界
    '''
    
    result_str, counts = re.subn(ip_pattern, "|||IP_ADDRESS|||", text, flags=re.VERBOSE)
    return result_str, counts

def run_classify_nsfw_implementation(text:str) -> tuple[Any,float]:
    label, score = ft_nsfw.predict(text)
    ret_label = ""
    if label[0] == "__label__non-nsfw":
        ret_label = "non-nsfw"
    else:
        ret_label = "nsfw"
    return ret_label, score[0]

def run_classify_toxic_speech_implementation(text:str) -> tuple[Any,float]:
    label, score = ft_toxic.predict(text)
    ret_label = ""
    print(label[0])
    if label[0] == "__label__toxic":
        ret_label = "toxic"
    else:
        ret_label = "non-toxic"
    return ret_label, score[0]

# Contain less than 50 or more than 100,000 words.
# Have a mean word length outside the range of 3 to 10 characters.
# Have more than 30% of lines ending with an ellipsis (“...”).
# Contain less than 80% of words with at least one alphabetic character.
def run_gopher_quality_filter_implementation(text: str) -> bool:
    # Condition 1: Word count        
    words = re.findall(r'\b\w+\b', text)
    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False

    # Condition 2: Mean word length
    word_lengths = [len(word) for word in words]
    if word_lengths:  # Avoid ZeroDivisionError
        mean_length = sum(word_lengths) / len(word_lengths)
    else:
        mean_length = 0
    if not (3 <= mean_length <= 10):
        return False

    # Condition 3: More than 30% of lines ending with ellipsis
    lines = text.splitlines()
    if lines:  # Avoid ZeroDivisionError
        num_ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        ellipsis_ratio = num_ellipsis_lines / len(lines)
    else:
        ellipsis_ratio = 0
    if ellipsis_ratio > 0.3:
        return False

    # Condition 4: Less than 80% of words with at least one alphabetic character
    alpha_words = [word for word in words if re.search('[a-zA-Z]', word)]
    if num_words == 0:  # Avoid ZeroDivisionError
        alpha_ratio = 0
    else:
        alpha_ratio = len(alpha_words) / num_words
    if alpha_ratio < 0.8:
        return False

    # Passed all conditions
    return True

# use xopen if available so compressed files work; otherwise fall back to built-in open
try:
    from xopen import xopen  # type: ignore
except Exception:
    xopen = open  # type: ignore


def run_exact_line_deduplication_implementation(input_files: List[os.PathLike], output_directory: os.PathLike) -> Dict[str, Any]:
    """
    Two-pass exact-line deduplication that keeps only lines that are unique across the entire corpus.
    First pass: count frequencies of exact lines (using sha256 digest as bucket key to reduce top-level memory).
    Second pass: rewrite each input file into output_directory/<original_name>, keeping only lines whose
    corpus frequency == 1.

    Returns a summary dict with counts.
    """
    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)

    # map: sha256_digest (bytes) -> dict mapping exact line string -> count
    counts: Dict[bytes, Dict[str, int]] = {}

    total_input_lines = 0

    # First pass: count occurrences
    for p in input_files:
        pth = Path(p)
        with xopen(pth, mode="rt", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                total_input_lines += 1
                # preserve the exact read line (including newline) as the key for exact matching
                b = line.encode("utf-8")
                h = hashlib.sha256(b).digest()
                bucket = counts.get(h)
                if bucket is None:
                    counts[h] = {line: 1}
                else:
                    bucket[line] = bucket.get(line, 0) + 1

    # Precompute which exact lines are unique (count == 1)
    # We create a set of (hash, line) pairs for fast lookup in pass 2.
    unique_pairs: Dict[bytes, set] = {}
    total_unique_lines = 0
    for h, bucket in counts.items():
        uniqs = {line for line, c in bucket.items() if c == 1}
        if uniqs:
            unique_pairs[h] = uniqs
            total_unique_lines += len(uniqs)

    for p in input_files:
        pth = Path(p)
        out_path = outdir / pth.name
        with xopen(pth, mode="rt", encoding="utf-8", errors="replace") as fin, \
             xopen(out_path, mode="wt", encoding="utf-8", errors="replace") as fout:
            for line in fin:
                b = line.encode("utf-8")
                h = hashlib.sha256(b).digest()
                # if hash not seen at all, it's not unique; otherwise check exact membership
                if h in unique_pairs and line in unique_pairs[h]:
                    fout.write(line)

import os
import re
import random
import hashlib
import unicodedata
from typing import List, Set, Tuple, Dict
from collections import defaultdict

PRIME = (1 << 61) - 1  # large prime for hashing


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = ''.join(ch for ch in s if not unicodedata.category(ch).startswith('M'))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_ngrams(tokens: List[str], n: int) -> Set[str]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


class MinHash:
    def __init__(self, num_hashes: int, seed: int = 42):
        self.num_hashes = num_hashes
        random.seed(seed)
        self.a = [random.randrange(1, PRIME - 1) for _ in range(num_hashes)]
        self.b = [random.randrange(0, PRIME - 1) for _ in range(num_hashes)]

    def _hash_shingle(self, shingle: str) -> int:
        h = hashlib.sha1(shingle.encode('utf8')).digest()
        return int.from_bytes(h[:8], 'big') % PRIME

    def signature(self, shingles: Set[str]) -> List[int]:
        sig = [PRIME] * self.num_hashes
        for sh in shingles:
            x = self._hash_shingle(sh)
            for i in range(self.num_hashes):
                val = (self.a[i] * x + self.b[i]) % PRIME
                if val < sig[i]:
                    sig[i] = val
        return sig


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    if not sig1 or not sig2:
        return 0.0
    equal = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return equal / len(sig1)


def lsh_candidates(signatures: List[List[int]], num_bands: int) -> Set[Tuple[int, int]]:
    if not signatures:
        return set()
    num_hashes = len(signatures[0])
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be divisible by num_bands")
    rows_per_band = num_hashes // num_bands

    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    candidates = set()

    for doc_idx, sig in enumerate(signatures):
        for b in range(num_bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(sig[start:end])
            h = hashlib.sha1(str(band_tuple).encode('utf8')).hexdigest()
            key = (b, int(h[:16], 16))
            buckets[key].append(doc_idx)

    for doc_list in buckets.values():
        if len(doc_list) > 1:
            doc_list_sorted = sorted(doc_list)
            for i in range(len(doc_list_sorted)):
                for j in range(i + 1, len(doc_list_sorted)):
                    candidates.add((doc_list_sorted[i], doc_list_sorted[j]))
    return candidates


def connected_components(n_nodes: int, edges: Set[Tuple[int, int]]) -> List[Set[int]]:
    g = [[] for _ in range(n_nodes)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    seen = [False] * n_nodes
    comps = []
    for i in range(n_nodes):
        if not seen[i]:
            stack = [i]
            comp = set()
            while stack:
                cur = stack.pop()
                if seen[cur]:
                    continue
                seen[cur] = True
                comp.add(cur)
                for nb in g[cur]:
                    if not seen[nb]:
                        stack.append(nb)
            comps.append(comp)
    return comps


def run_minhash_deduplication(
    input_paths: List[str],
    num_hashes: int,
    num_bands: int,
    ngram: int,
    output_dir: str,
    similarity_threshold: float = 0.9,
) -> None:
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be divisible by num_bands")

    os.makedirs(output_dir, exist_ok=True)

    docs_raw: List[str] = []
    basenames: List[str] = []
    for p in input_paths:
        with open(p, 'r', encoding='utf8') as f:
            docs_raw.append(f.read())
        basenames.append(os.path.basename(p))

    docs_norm = [normalize_text(d) for d in docs_raw]
    docs_tokens = [d.split() for d in docs_norm]
    docs_ngrams = [word_ngrams(toks, ngram) for toks in docs_tokens]

    mh = MinHash(num_hashes, seed=seed)
    signatures = [mh.signature(sh) for sh in docs_ngrams]

    candidates = lsh_candidates(signatures, num_bands)

    duplicate_edges: Set[Tuple[int, int]] = set()
    for i, j in candidates:
        sim = minhash_similarity(signatures[i], signatures[j])
        if sim >= similarity_threshold:
            duplicate_edges.add((i, j))

    comps = connected_components(len(input_paths), duplicate_edges)

    random.seed(seed)
    keep = [False] * len(input_paths)
    for comp in comps:
        if len(comp) == 1:
            idx = next(iter(comp))
            keep[idx] = True
        else:
            chosen = random.choice(sorted(list(comp)))
            keep[chosen] = True

    for idx, should_keep in enumerate(keep):
        out_path = os.path.join(output_dir, basenames[idx])
        if should_keep:
            with open(out_path, 'w', encoding='utf8') as f:
                f.write(docs_raw[idx])
