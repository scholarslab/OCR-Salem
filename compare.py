#!/usr/bin/env python3
"""Compare OCR output against ground truth text."""

import difflib
import re
import unicodedata
import string
from pathlib import Path

from pathlib import Path

DIR = './ecca2089r/'

# Get ocr files in DIR
OCR_FILES = [Path(DIR+item.name) for item in Path(DIR).iterdir() if item.is_file() and item.name.startswith("ocr-")]

GROUND_TRUTH_FILE = Path(DIR+"gt.txt")

# Remove formatting syntax (LaTeX, Markdown) before comparison
STRIP_FORMATTING = True

# Remove punctuation, alphanumerics only.
STRIP_PUNCTUATION = True

# Ignore casing or not?
IGNORE_CASING = True

# Normalize unicode characters (sub/superscript, accented latin) to ASCII
NORMALIZE_UNICODE = True

def strip_formatting(text: str) -> str:
    """Remove LaTeX and Markdown formatting syntax, keeping content."""
    result = text
    # Special case: remove strikethroughs entirely
    result = re.sub(r"~~(.+?)~~","", result)
    # Otherwise: deformat LaTeX math mode: $...$ or $$...$$
    result = re.sub(r"\$\$(.+?)\$\$", r"\1", result, flags=re.DOTALL)
    result = re.sub(r"\$(.+?)\$", r"\1", result)
    # ... and LaTeX commands: \command{arg} -> arg, \command -> ""
    result = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", result)
    result = re.sub(r"\\[a-zA-Z]+", "", result)
    # ... and LaTeX braces used for grouping
    result = re.sub(r"(?<!\\)[{}]", "", result)
    # ... and Markdown headers: # Header -> Header
    result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)
    # ... and Markdown bold/italic: **text** -> text, *text* -> text
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    result = re.sub(r"\*(.+?)\*", r"\1", result)
    result = re.sub(r"__(.+?)__", r"\1", result)
    result = re.sub(r"_(.+?)_", r"\1", result)
    # Markdown inline code: `code` -> code
    result = re.sub(r"`([^`]+)`", r"\1", result)
    # Markdown links: [text](url) -> text
    result = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", result)

    return result

def strip_punctuation(text: str) -> str:
    """Remove all punctuation """
    return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents.

    Uses NFKD normalization which handles sub/superscripts, accented
    characters, and other compatibility variants.
    """
    result = unicodedata.normalize("NFKD", text)
    return result.encode("ascii", "ignore").decode("ascii")


def levenshtein_distance(s1, s2) -> int:
    """Calculate the Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(ground_truth: str, ocr_output: str) -> float:
    """Calculate Character Error Rate (CER)."""
    distance = levenshtein_distance(ground_truth, ocr_output)
    return distance / len(ground_truth) if ground_truth else 0.0


def word_error_rate(ground_truth: str, ocr_output: str) -> float:
    """Calculate Word Error Rate (WER)."""
    gt_words = ground_truth.split()
    ocr_words = ocr_output.split()
    distance = levenshtein_distance(tuple(gt_words), tuple(ocr_words))
    return distance / len(gt_words) if gt_words else 0.0


def compare_texts(ground_truth: str, ocr_output: str) -> dict:
    """Compare ground truth with OCR output and return metrics."""
    gt_normalized = ground_truth.strip()
    ocr_normalized = ocr_output.strip()

    if STRIP_FORMATTING:
        gt_normalized = strip_formatting(gt_normalized)
        ocr_normalized = strip_formatting(ocr_normalized)

    if NORMALIZE_UNICODE:
        gt_normalized = normalize_unicode(gt_normalized)
        ocr_normalized = normalize_unicode(ocr_normalized)
    
    if STRIP_PUNCTUATION:
        gt_normalized = strip_punctuation(gt_normalized)
        ocr_normalized = strip_punctuation(ocr_normalized)
    
    if IGNORE_CASING:
        gt_normalized = gt_normalized.upper()
        ocr_normalized = ocr_normalized.upper()
    
    # minimize whitespace
    gt_normalized = " ".join(gt_normalized.strip().split())
    ocr_normalized = " ".join(ocr_normalized.strip().split())
    
    cer = character_error_rate(gt_normalized, ocr_normalized)
    wer = word_error_rate(gt_normalized, ocr_normalized)
    similarity = difflib.SequenceMatcher(None, gt_normalized, ocr_normalized).ratio()

    return {
        "character_error_rate": cer,
        "word_error_rate": wer,
        "similarity_ratio": similarity,
        "accuracy": 1 - cer,
        "ground_truth_chars": len(gt_normalized),
        "ocr_chars": len(ocr_normalized),
        "ground_truth_words": len(gt_normalized.split()),
        "ocr_words": len(ocr_normalized.split()),
    }


def print_comparison(name: str, metrics: dict):
    """Print comparison metrics in a readable format."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Character Error Rate (CER): {metrics['character_error_rate']:.2%}")
    print(f"  Word Error Rate (WER):      {metrics['word_error_rate']:.2%}")
    print(f"  Similarity Ratio:           {metrics['similarity_ratio']:.2%}")
    print(f"  Character Accuracy:         {metrics['accuracy']:.2%}")
    print(f"  Ground Truth: {metrics['ground_truth_chars']} chars, {metrics['ground_truth_words']} words")
    print(f"  OCR Output:   {metrics['ocr_chars']} chars, {metrics['ocr_words']} words")


def main():
    ground_truth = GROUND_TRUTH_FILE.read_text(encoding="utf-8")

    print(f"Ground Truth: {GROUND_TRUTH_FILE}")
    print(f"Characters: {len(ground_truth.strip())}, Words: {len(ground_truth.split())}")

    results = []
    for ocr_file in OCR_FILES:
        if not ocr_file.exists():
            print(f"\nSkipping {ocr_file} (not found)")
            continue

        ocr_text = ocr_file.read_text(encoding="utf-8")
        metrics = compare_texts(ground_truth, ocr_text)
        print_comparison(ocr_file.name, metrics)
        results.append((ocr_file.name, metrics))

    if results:
        print(f"\n{'=' * 50}")
        print("  SUMMARY (sorted by accuracy)")
        print(f"{'=' * 50}")
        results.sort(key=lambda x: x[1]["accuracy"], reverse=True)
        for name, m in results:
            print(f"  {name:<25} CER: {m['character_error_rate']:>6.2%}  Accuracy: {m['accuracy']:>6.2%}")


if __name__ == "__main__":
    main()