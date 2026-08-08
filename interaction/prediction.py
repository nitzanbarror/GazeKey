"""GazeKey — offline word prediction for the suggestion bar.

Suggestions must be **instant and local** (spec NFR-5): nothing about what the
user types leaves the machine, and nothing is looked up over a network.

``pyenchant`` is deliberately not used. It is a *spell checker*: it can tell you
whether a word exists but not which of several candidates is more likely, so it
cannot rank suggestions — and its Windows builds need external enchant DLLs
that frequently do not resolve. A small bundled frequency list does the job
with no dependency at all.

Ranking blends two sources:

* the bundled list, ordered by rough English frequency;
* a **personal count** of words this user has actually chosen, persisted to
  ``~/.gazekey/user_words.json``. A word the user picks repeatedly climbs above
  more common English words, which is the whole point for someone typing the
  same names and phrases every day.

The exact prefix, if it is itself a word, is not offered back: the Space key
already completes it, so the slot is better spent on a real completion.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

#: Common English words in rough frequency order. Deliberately small: it is
#: enough to make the suggestion bar useful without shipping a corpus, and the
#: personal counts below are what make it fit an individual user.
COMMON_WORDS = """
the be to of and a in that have i it for not on with he as you do at this but
his by from they we say her she or an will my one all would there their what
so up out if about who get which go me when make can like time no just him know
take people into year your good some could them see other than then now look
only come its over think also back after use two how our work first well way
even new want because any these give day most us is are was were been has had
said did going made find where much before right too old any same tell does set
yes no here hello hey hi ok okay please thanks sorry help stop wait more again
three want does put end why let great man thing woman life child world school
state family student group country problem hand part place case week company
system program question work government number night point home water room
mother area money story fact month lot book eye job word business issue side
kind head house service friend father power hour game line end member law car
city community name president team minute idea kid body information back parent
face others level office door health person art war history party result change
morning reason research girl guy moment air teacher force education foot boy age
policy process music market sense nation plan college interest death course
someone experience behind reach local kill six remain effect use yes yeah hello
hey hi thanks thank please sorry okay ok maybe sure today tonight tomorrow
yesterday morning evening afternoon here there everywhere something anything
nothing everything everyone somebody nobody always never sometimes often
usually really very quite almost enough little more less many few much most
best better worse worst high low long short big small large great good bad
happy sad tired hungry thirsty hot cold warm cool nice fine well sick pain help
need want love hate hope wish feel think know understand remember forget
believe mean try keep leave stay move stop start begin finish open close read
write speak talk listen hear watch look show tell ask answer call send bring
carry hold turn walk run sit stand sleep wake eat drink wash dress please
water food drink coffee tea bread milk juice medicine doctor nurse hospital
phone message email family friend wife husband son daughter brother sister
mother father grandmother grandfather please thank you sorry excuse morning
afternoon evening night today tomorrow yesterday minute hour day week month
year time clock alarm light dark window door bed chair table kitchen bathroom
outside inside upstairs downstairs television radio music computer keyboard
screen battery charger blanket pillow warm cold comfortable uncomfortable
please turn switch raise lower louder quieter brighter darker slower faster
yourself myself himself herself themselves ourselves together alone ready
finished waiting looking talking eating drinking sleeping sitting standing
walking reading writing thinking feeling breathing hurting itching burning
better worse okay alright certainly definitely probably possibly absolutely
tomorrow tonight later earlier soon already still yet almost nearly hardly
question answer problem solution message letter number address birthday
appointment visit visitor guest neighbour neighbor colleague company meeting
breakfast lunch dinner supper snack sandwich soup salad fruit vegetable
"""


def _load_bundled() -> Dict[str, int]:
    """Word -> rank (lower is more common), first occurrence wins."""
    ranks: Dict[str, int] = {}
    for index, word in enumerate(COMMON_WORDS.split()):
        ranks.setdefault(word.lower(), index)
    return ranks


BUNDLED_RANKS: Dict[str, int] = _load_bundled()

#: how much one personal use is worth, in "rank places climbed"
PERSONAL_WEIGHT = 400.0

#: personal-only words start here, so a word the user invented still competes
UNRANKED_BASE = len(BUNDLED_RANKS) + 500

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class WordPredictor:
    """Ranks completions of a prefix from a bundled list plus personal use."""

    def __init__(
        self,
        path: Optional[str] = None,
        extra_words: Optional[Iterable[str]] = None,
        autosave: bool = True,
    ) -> None:
        self.path = path
        self.autosave = autosave
        self._ranks: Dict[str, int] = dict(BUNDLED_RANKS)
        for index, word in enumerate(extra_words or ()):
            self._ranks.setdefault(word.lower(), UNRANKED_BASE + index)
        self._personal: Dict[str, int] = defaultdict(int)
        self._index: Dict[str, List[str]] = {}
        self._load_personal()
        self._rebuild_index()

    # ------------------------------------------------------------------ index
    def _rebuild_index(self) -> None:
        """Bucket every word by its first three letters for instant lookup."""
        index: Dict[str, List[str]] = defaultdict(list)
        for word in set(self._ranks) | set(self._personal):
            for size in (1, 2, 3):
                if len(word) >= size:
                    index[word[:size]].append(word)
        self._index = {key: sorted(words, key=self.score)
                       for key, words in index.items()}

    def score(self, word: str) -> float:
        """Lower is better: bundled rank pulled down by personal use."""
        base = float(self._ranks.get(word, UNRANKED_BASE + len(word)))
        return base - PERSONAL_WEIGHT * self._personal.get(word, 0)

    # ------------------------------------------------------------- suggestions
    def suggest(self, prefix: str, limit: int = 4) -> List[str]:
        """Best completions of ``prefix``, most likely first."""
        prefix = (prefix or "").strip().lower()
        if not prefix:
            return []
        bucket = self._index.get(prefix[:3]) if len(prefix) >= 3 else \
            self._index.get(prefix[:len(prefix)])
        candidates = bucket if bucket is not None else []
        matches = [word for word in candidates
                   if word.startswith(prefix) and word != prefix]
        matches.sort(key=self.score)
        return matches[:limit]

    def completion_for(self, prefix: str, word: str) -> str:
        """The letters still to type to turn ``prefix`` into ``word``."""
        if word.lower().startswith(prefix.lower()):
            return word[len(prefix):]
        return word

    # ---------------------------------------------------------------- learning
    def record(self, word: str) -> None:
        """Remember that the user chose this word."""
        word = (word or "").strip().lower()
        if not word or not WORD_RE.fullmatch(word):
            return
        known = word in self._ranks or word in self._personal
        self._personal[word] += 1
        if not known:
            self._ranks.setdefault(word, UNRANKED_BASE)
        self._rebuild_index()
        if self.autosave:
            self.save()

    def personal_count(self, word: str) -> int:
        return self._personal.get(word.lower(), 0)

    # ------------------------------------------------------------- persistence
    def _load_personal(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(stored, dict):
            for word, count in stored.items():
                if isinstance(word, str) and isinstance(count, int) and count > 0:
                    self._personal[word.lower()] = count
                    self._ranks.setdefault(word.lower(), UNRANKED_BASE)

    def save(self) -> None:
        """Persist the personal counts (words only — never sentences)."""
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(dict(self._personal), handle, indent=2,
                          ensure_ascii=False)
        except OSError:
            pass          # a read-only home must not break typing


class WordBuffer:
    """Tracks the word currently being typed, for the suggestion bar."""

    def __init__(self) -> None:
        self.text = ""

    def add(self, char: str) -> None:
        if not char:
            return
        if WORD_RE.fullmatch(char):
            self.text += char
        else:
            self.text = ""          # punctuation ends the word

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""

    def __bool__(self) -> bool:
        return bool(self.text)
