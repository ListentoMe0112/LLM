import json
from typing import List, Dict, Optional, Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens = sorted(self.special_tokens, key = lambda x : -len(x))

        # Add special tokens to vocab if they aren't already present
        for token in self.special_tokens:
            if token.encode("utf-8") not in self.vocab.values():
                new_id = len(self.vocab)
                self.vocab[new_id] = token.encode("utf-8")

        self.vocab_inv = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> 'Tokenizer':
        vocab = {}
        with open(vocab_filepath) as vocab_f:
            vocab = json.load(vocab_f)
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        # vocab = {
        #     gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        #     for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        # }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str, max_workers: int = 4) -> List[int]:
        if text == "":
            return []
        
        # Split the text into documents and special tokens
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        pattern = "|".join(escaped_tokens)  # Create a pattern to match any special token
        parts = re.split(f"({pattern})", text)

        # Reassemble into separate documents (chunks) and collect special tokens
        documents = []
        split_tokens = []
        current_doc = []
        for part in parts:
            if part in self.special_tokens:
                if current_doc:  # Submit the current document
                    documents.append("".join(current_doc))
                    split_tokens.append(part)
                    current_doc = []
            else:
                current_doc.append(part)
        if current_doc:
            documents.append("".join(current_doc))

        # Process documents in parallel
        tokens = []
        if max_workers > 1 and len(documents) > 1:
            # Use multiprocessing for multiple documents
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Process each document in parallel
                futures = []
                for doc in documents:
                    futures.append(executor.submit(
                        self._encode_document, 
                        doc, 
                        self.merges, 
                        self.vocab_inv
                    ))
                
                # Collect results in order
                for i, future in enumerate(futures):
                    tokens.extend(future.result())
                    # Add special token after each document except the last
                    if i < len(split_tokens):
                        tokens.append(self.vocab_inv.get(split_tokens[i].encode("utf-8")))
        else:
            # Single-threaded processing
            for i, doc in enumerate(documents):
                tokens.extend(self._encode_document(doc, self.merges, self.vocab_inv))
                if i < len(split_tokens):
                    tokens.append(self.vocab_inv.get(split_tokens[i].encode("utf-8")))
                    
        return tokens

    @staticmethod
    def _encode_document(document: str, merges: List[tuple[bytes, bytes]], vocab_inv: Dict[bytes, int]) -> List[int]:
        """Process a single document and return token IDs"""
        tokens = []
        if not document:
            return tokens
            
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(pat, document):
            token_str = match.group()
            token_bytes = token_str.encode("utf-8")
            modified = True
            old_token_bytes = [bytes([b]) for b in token_bytes]
            while modified:
                modified = False
                for pair in merges:
                    i = 0
                    while i < len(old_token_bytes) - 1:
                        if (old_token_bytes[i], old_token_bytes[i+1]) == pair:
                            old_token_bytes[i] += old_token_bytes[i+1]
                            del old_token_bytes[i+1]
                            modified = True
                            break
                        i += 1
                    if modified:
                        break
            tokens.extend([vocab_inv.get(k) for k in old_token_bytes])
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        special_pattern = re.compile("(" + "|".join(map(re.escape, self.special_tokens)) + ")") if self.special_tokens else None
        word_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        for chunk in iterable:
            # Combine buffer with new chunk
            text = buffer + chunk
            if not text:
                continue

            # Split text into parts using special tokens (same logic as encode())
            parts = []
            if special_pattern:
                parts = special_pattern.split(text)
            else:
                parts = [text]

            # Reassemble into documents and special tokens (matching encode() logic)
            current_doc = []
            pending_specials = []
            for i, part in enumerate(parts):
                if special_pattern and i % 2 == 1:  # Special token
                    if current_doc:
                        # Process accumulated document content first
                        doc_text = "".join(current_doc)
                        matches = list(word_pat.finditer(doc_text))
                        if matches:
                            last_valid_end = matches[-1].end()
                            yield from self._process_document(doc_text[:last_valid_end], word_pat)
                            buffer = doc_text[last_valid_end:]
                        else:
                            buffer = doc_text
                        current_doc = [buffer]
                    
                    # Immediately process special token to maintain order
                    yield self.vocab_inv[part.encode("utf-8")]
                else:
                    current_doc.append(part)

            # Process remaining document content after splitting
            doc_text = "".join(current_doc)
            if doc_text:
                matches = list(word_pat.finditer(doc_text))
                if matches:
                    last_valid_end = matches[-1].end()
                    yield from self._process_document(doc_text[:last_valid_end], word_pat)
                    buffer = doc_text[last_valid_end:]
                else:
                    buffer = doc_text

            # Handle pending special tokens
            for special in pending_specials:
                yield self.vocab_inv[special.encode("utf-8")]
            pending_specials.clear()
                
        # Process remaining buffer
        if buffer:
            yield from self._process_document(buffer, word_pat)

    def _process_document(self, text: str, word_pat: re.Pattern) -> Iterator[int]:
        """Process a single document segment (non-special token content)"""
        for match in word_pat.finditer(text):
            yield from self._process_token(match.group())
    
    def _process_token(self, token_str: str) -> Iterator[int]:
        """Helper method to process individual tokens"""
        token_bytes = token_str.encode("utf-8")
        old_token_bytes = [bytes([b]) for b in token_bytes]
        
        # Apply BPE merges
        modified = True
        while modified:
            modified = False
            for pair in self.merges:
                i = 0
                while i < len(old_token_bytes) - 1:
                    if (old_token_bytes[i], old_token_bytes[i+1]) == pair:
                        old_token_bytes[i] += old_token_bytes[i+1]
                        del old_token_bytes[i+1]
                        modified = True
                        break  # Restart after modification
                    i += 1
                if modified:
                    break
        
        # Yield token IDs
        for k in old_token_bytes:
            yield self.vocab_inv.get(k, self.vocab_inv.get(b"", 0))

    def decode(self, ids: List[int]) -> str:
        # Decode token IDs into text
        bytes_list = []
        if len(ids) == 0:
            return ""
        for id in ids:
            bytes_list.append(self.vocab.get(id))
        final_bytes = b''.join(bytes_list)
        return final_bytes.decode('utf-8', errors='replace') 

