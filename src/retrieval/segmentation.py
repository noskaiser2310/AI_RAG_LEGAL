import functools
import logging

logger = logging.getLogger(__name__)

_underthesea_loaded = False


def _get_tokenizer():
    global _underthesea_loaded
    if not _underthesea_loaded:
        logger.info("Loading underthesea Vietnamese tokenizer...")
    _underthesea_loaded = True
    from underthesea import word_tokenize
    return word_tokenize


@functools.lru_cache(maxsize=50000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    tokenizer = _get_tokenizer()
    return tuple(tokenizer(text))


def tokenize_vietnamese(text: str) -> list[str]:
    return list(_tokenize_cached(text))


def is_vietnamese_tokenizer_available() -> bool:
    try:
        _get_tokenizer()
        return True
    except Exception:
        return False


def clear_token_cache():
    _tokenize_cached.cache_clear()
