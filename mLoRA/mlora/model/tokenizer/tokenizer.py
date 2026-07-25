import logging
from typing import Tuple

from transformers import AutoTokenizer

from mlora.model.args import Masks, Tokens

logger = logging.getLogger(__name__)

# Running tally of prompts that hit the cutoff_len cap, so a silent truncation
# cannot pass unnoticed. See the warning in Tokenizer.encode below.
TRUNCATION_STATS = {"encoded": 0, "truncated": 0, "max_seen": 0}


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer_ = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.n_words_ = self.tokenizer_.vocab_size
        self.bos_id_ = self.tokenizer_.bos_token_id
        self.eos_id_ = self.tokenizer_.eos_token_id
        self.pad_id_ = self.tokenizer_.pad_token_id
        self.unk_id_ = self.tokenizer_.unk_token_id
        # maybe pad id is unk
        if self.pad_id_ is None:
            assert self.unk_id_ is not None or self.eos_id_ is not None
            self.pad_id_ = self.unk_id_ if self.unk_id_ is not None else self.eos_id_

    def encode(self, data: str, bos=True, eos=True, cutoff_len=4096) -> Tokens:
        tokens = self.tokenizer_.encode(data, add_special_tokens=False)
        # Qwen and others have bos_token_id / eos_token_id = None; never insert None.
        use_bos = bos and self.bos_id_ is not None
        use_eos = eos and self.eos_id_ is not None

        # ── Truncation detector (observability only, behaviour unchanged) ──
        # cutoff_len defaults to 4096 and callers do not override it. In the
        # UG-TTT prompt templates the <<<LAST_CODE>>> placeholder sits near the
        # END, so everything truncation removes first is the task instruction,
        # the Rules, and the "return the final program between ```python and
        # ```" line that last_codeblock_postprocess depends on -- which makes
        # the rollout score exactly 0.0 with no other symptom.
        #
        # Programs carried forward grow over a run, so this can switch on
        # mid-run. Long-program domains (ahc*/gpu_mode: 17-35 KB artifacts) are
        # far more exposed than the math domains.
        limit = cutoff_len - int(use_bos) - int(use_eos)
        TRUNCATION_STATS["encoded"] += 1
        TRUNCATION_STATS["max_seen"] = max(TRUNCATION_STATS["max_seen"], len(tokens))
        if len(tokens) > limit:
            TRUNCATION_STATS["truncated"] += 1
            logger.warning(
                "PROMPT TRUNCATED: %d tokens -> %d (cutoff_len=%d). The tail of "
                "the prompt, including the output-format instruction, was "
                "discarded; this rollout will most likely score 0. "
                "(%d/%d prompts truncated so far)",
                len(tokens), limit, cutoff_len,
                TRUNCATION_STATS["truncated"], TRUNCATION_STATS["encoded"],
            )

        tokens = tokens[:limit]
        if use_bos:
            tokens = [self.bos_id_] + tokens
        if use_eos:
            tokens = tokens + [self.eos_id_]
        return tokens

    def decode(self, data: Tokens) -> str:
        return self.tokenizer_.decode(data)

    @property
    def __padding_side(self) -> str:
        return self.tokenizer_.padding_side

    def expand_tokens(self, tokens: Tokens, align_len: int) -> Tuple[Tokens, Masks]:
        # get the mask from tokens, if True, will mask those token
        masks = [False] * len(tokens)
        if len(tokens) >= align_len:
            return tokens, masks

        pad_tokens = [self.pad_id_] * (align_len - len(tokens))
        pad_masks = [True] * (align_len - len(tokens))
        if self.__padding_side == "right":
            return tokens + pad_tokens, masks + pad_masks

        return pad_tokens + tokens, pad_masks + masks
