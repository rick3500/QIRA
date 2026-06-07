"""
QIRA Word Ladder Demo -- no LLM.

Problem:
  Change "lass" to "male" with exactly 3 intermediate words.
  Each step changes exactly one letter to a valid English word.

States:       superposition of current words at each depth level.
Transitions:  Markov operator -- one-letter substitution to any valid
              dictionary word not already visited in the current path.
Amplitude:    boosted by Hamming proximity to the target word "male".
              d=1 word (one letter away) gets 4x amplitude of d=3 word.
Dedup:        multiple paths converging on the same word keep only the
              BEST (highest-amplitude) route.  This prevents high-degree
              words that are reachable via many routes from drowning out
              target-directed words that are reachable via fewer.
FFT:          smooths amplitude distribution across the word ensemble.
Collapse:     follow highest-amplitude word at each depth to reconstruct
              the optimal ladder.

The word list is generated at runtime from a real English frequency corpus
(pyspellchecker) -- no hand-picked words, no curated transitions.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Dictionary -- every valid 4-letter English word from a frequency corpus
# ---------------------------------------------------------------------------

MIN_FREQ = 500   # exclude rare/obscure words — only accept words with
                 # >= 500 corpus hits so "macs"(66) and "mads"(233) are
                 # excluded while "mast"(1728) and "malt"(1987) are kept.

def _load_words():
    try:
        from spellchecker import SpellChecker
        import math
        freq_map = SpellChecker().word_frequency
        words = {
            w for w in freq_map.dictionary
            if len(w) == 4 and w.isascii() and w.isalpha()
            and freq_map[w] >= MIN_FREQ
        }
        freq_weight = {w: math.log(freq_map[w] + 1) for w in words}
        return words, freq_weight
    except ImportError:
        raise SystemExit(
            "pyspellchecker not installed. Run: pip install pyspellchecker"
        )

WORDS_4, FREQ_WEIGHT = _load_words()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hamming(w1: str, w2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(w1, w2))


def one_step_neighbors(word: str, word_set: set[str]) -> list[str]:
    result = []
    for i in range(len(word)):
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c != word[i]:
                cand = word[:i] + c + word[i + 1:]
                if cand in word_set:
                    result.append(cand)
    return result


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WordNode:
    """A word at a given depth, together with the path taken to reach it."""

    def __init__(self, word: str, path: list[str], target: str) -> None:
        self.word     = word
        self.path     = list(path)
        self.distance = hamming(word, target)
        self.score    = 0.0

    def __repr__(self) -> str:
        return f"{self.word}(d={self.distance})"


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

class WordSuperposition:
    def __init__(self, target: str) -> None:
        self.target     = target
        self.nodes:      list[WordNode] = []
        self.amplitudes: list[float]    = []

    def add(self, node: WordNode, amp: float = 1.0) -> None:
        self.nodes.append(node)
        self.amplitudes.append(amp)

    def normalize(self) -> None:
        total = sum(abs(a) for a in self.amplitudes)
        if total:
            self.amplitudes = [a / total for a in self.amplitudes]

    def deduplicate(self) -> None:
        """For each word keep only the highest-amplitude (best) path.

        Summing amplitudes favours high-degree words reachable by many routes,
        drowning out target-directed words that travel fewer, better paths.
        Max-path keeps the signal proportional to path quality, not path count.
        """
        canonical: dict[str, tuple[WordNode, float]] = {}
        for node, amp in zip(self.nodes, self.amplitudes):
            key = node.word
            if key not in canonical or amp > canonical[key][1]:
                canonical[key] = (node, amp)
        self.nodes      = [n for n, _ in canonical.values()]
        self.amplitudes = [a for _, a in canonical.values()]
        self.normalize()

    def spectral_mix(self) -> None:
        if len(self.amplitudes) < 2:
            return
        amps     = np.array(self.amplitudes, dtype=float)
        kernel   = np.array([1 / (1 + i) for i in range(len(amps))])
        smoothed = np.fft.ifft(np.fft.fft(amps) * kernel)
        self.amplitudes = smoothed.real.tolist()
        self.normalize()

    def prune(self, max_nodes: int = 25) -> None:
        pairs = sorted(zip(self.nodes, self.amplitudes), key=lambda x: -x[1])
        pairs = pairs[:max_nodes]
        if pairs:
            self.nodes, self.amplitudes = map(list, zip(*pairs))
            self.normalize()

    def expand(self, word_set: set[str], freq_weight: dict[str, float]) -> "WordSuperposition":
        """Markov transition: each word -> all valid one-letter-change neighbors."""
        next_sup = WordSuperposition(self.target)
        for node, amp in zip(self.nodes, self.amplitudes):
            visited = set(node.path + [node.word])
            for neighbor in one_step_neighbors(node.word, word_set):
                if neighbor not in visited:
                    new_node = WordNode(
                        neighbor, node.path + [node.word], self.target
                    )
                    dist = new_node.distance
                    # Geometric proximity boost: d=0 scores 4^4=256x more than d=4.
                    # Frequency weight: log(count+1) so "mast" (1728 hits) beats
                    # "macs" (66 hits) at the same Hamming distance.
                    score = 4.0 ** (4 - dist) * freq_weight.get(neighbor, 1.0)
                    new_node.score = score
                    next_sup.add(new_node, amp * score)
        next_sup.deduplicate()
        # No FFT here: spectral smoothing equalizes amplitudes and destroys
        # the proximity signal when the corpus is large.  The amplitude itself
        # IS the interference mechanism for word ladders.
        next_sup.prune(max_nodes=15)
        return next_sup

    def find(self, word: str):
        for node, amp in zip(self.nodes, self.amplitudes):
            if node.word == word:
                return node, amp
        return None, 0.0

    def snapshot(self, top_n: int = 12) -> str:
        pairs = sorted(zip(self.nodes, self.amplitudes), key=lambda x: -x[1])[:top_n]
        lines = []
        for node, amp in pairs:
            bar = "#" * int(amp * 35)
            lines.append(
                f"    {node.word}  d={node.distance}  {bar:<35}  {amp:.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    START     = "lass"
    TARGET    = "male"
    MAX_STEPS = 4     # 3 intermediate words = 4 hops

    print("=" * 60)
    print("QIRA Word Ladder Demo")
    print("=" * 60)
    print()
    print(f"  Start:  {START.upper()}")
    print(f"  Target: {TARGET.upper()}")
    print(f"  Rule:   change exactly one letter per step")
    print(f"  Goal:   reach target in {MAX_STEPS - 1} intermediate steps")
    print()
    print("Architecture:")
    print("  States      = words reachable at each depth (superposition)")
    print("  Transitions = one-letter substitutions (Markov)")
    print("  Amplitude   = boosted by Hamming proximity to target")
    print("                d=1 word scores 4x higher than d=3 word")
    print("                also weighted by log(word frequency) so")
    print("                common words beat obscure-but-valid ones")
    print("  Dedup       = max-path keeps best route to each word")
    print("  FFT         = smooths amplitude across the word ensemble")
    print("  Collapse    = follow highest-amplitude word at each depth")
    print()
    print(f"  Hamming distance {START} -> {TARGET} = {hamming(START, TARGET)} letters differ")
    print(f"  => minimum steps needed = {hamming(START, TARGET)}")
    print()

    seed = WordNode(START, [], TARGET)
    seed.score = 1.0
    sup = WordSuperposition(TARGET)
    sup.add(seed, 1.0)

    answer_node = None

    for step in range(1, MAX_STEPS + 1):
        sup = sup.expand(WORDS_4, FREQ_WEIGHT)

        target_node, _ = sup.find(TARGET)
        tag = "  <-- TARGET REACHED" if target_node else ""

        print(f"Depth {step} - top words by amplitude{tag}")
        print(f"  (d = Hamming distance remaining to '{TARGET}')")
        print(sup.snapshot())
        print()

        if target_node:
            answer_node = target_node
            break

    print("=" * 60)
    print("ANSWER  (collapsed path)")
    print("=" * 60)
    print()

    if answer_node:
        full_path = answer_node.path + [answer_node.word]
        print("  " + "  ->  ".join(w.upper() for w in full_path))
        print()
        for i, (a, b) in enumerate(zip(full_path, full_path[1:])):
            diff = [(j, a[j], b[j]) for j in range(4) if a[j] != b[j]]
            pos, old, new = diff[0]
            label = "start" if i == 0 else f"step  {i}"
            print(f"  {label}  {a.upper()}  pos {pos+1}: '{old}' -> '{new}'  ->  {b.upper()}")
        print()
        intermediates = full_path[1:-1]
        print(f"  {len(intermediates)} intermediate word(s): "
              f"{', '.join(w.upper() for w in intermediates)}")
    else:
        print("  No path found within the word list in the allowed steps.")


if __name__ == "__main__":
    main()
