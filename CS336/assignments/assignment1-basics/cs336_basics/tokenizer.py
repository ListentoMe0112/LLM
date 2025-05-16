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

    def encode(self, text: str) -> List[int]:
        tokens = []
        if text == "":
            return tokens
 
        # Encode the text into a list of token IDs after applying BPE
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        pattern = "|".join(escaped_tokens)  # Create a pattern to match any special token
        # split_indices = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

        # Split the text at the special tokens, keeping them as separators
        parts = re.split(f"({pattern})", text)

        # Reassemble into separate documents (chunks), skipping the special tokens
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

        # Add the last document if there is remaining content
        if current_doc:
            documents.append("".join(current_doc))

        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_token_idx = 0

        for single_text in documents:
            for match in re.finditer(pat, single_text):
                token_str = match.group()
                token_bytes = token_str.encode("utf-8")
                modified = True
                old_token_bytes = list(bytes([b]) for b in token_bytes)
                while modified:
                    modified = False
                    # Iterate over the list of merges
                    for pair in self.merges:
                        i = 0
                        # Check for the pair in the list of tokens
                        while i < len(old_token_bytes) - 1:
                            if (old_token_bytes[i], old_token_bytes[i + 1]) == pair:
                                # Merge the tokens
                                old_token_bytes[i] = old_token_bytes[i] + old_token_bytes[i + 1]
                                del old_token_bytes[i + 1]
                                modified = True  # A merge happened, so we need to restart the loop
                                break  # After a merge, restart the merge process from the beginning
                            i += 1
                        if modified:
                            break

                tokens += [self.vocab_inv.get(k) for k in old_token_bytes]

            if special_token_idx < len(split_tokens):
                tokens.append(self.vocab_inv.get(split_tokens[special_token_idx].encode("utf-8")))
                special_token_idx += 1

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
        print("New Word", token_str)
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

