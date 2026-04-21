"""
Speech Understanding - Programming Assignment 2
Part II: Phonetic Mapping & Translation

Tasks:
    2.1 - IPA Unified Representation of code-switched (Hinglish) transcript
          Custom word-level language router + G2P mapping layer for Hinglish phonology
    2.2 - Semantic Translation into Bhojpuri (target LRL)
          500-word technical parallel corpus for speech/physics terms

Folder structure expected:
    /scratch/data/m22aie221/workspace/CSL7770-Assignment2/
        data/
            lecture_segment.wav
        part1/
            outputs/
                transcript.json          ← from Part 1
                lid_predictions.json     ← from Part 1
        part2/
            part2_phonetic_translation.py   ← this file
            corpus/
                hinglish_bhojpuri_corpus.json   ← auto-generated
            outputs/
                ipa_transcript.json
                ipa_transcript.txt
                bhojpuri_translation.json
                bhojpuri_translation.txt
                part2_metrics.json

Usage:
    python part2_phonetic_translation.py --transcript ../part1/outputs/transcript.json
    python part2_phonetic_translation.py --transcript ../part1/outputs/transcript.json --mode ipa
    python part2_phonetic_translation.py --transcript ../part1/outputs/transcript.json --mode translate
    python part2_phonetic_translation.py --transcript ../part1/outputs/transcript.json --mode full
"""

import os
import re
import json
import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TASK 2.2 CORPUS  — 500-word Hinglish↔Bhojpuri parallel corpus
# (Technical terms from speech processing + quantum physics,
#  matching HC Verma lecture content)
# ─────────────────────────────────────────────────────────────

# Format: "hindi/english_term": ("bhojpuri_translation", "ipa_bhojpuri")
# Bhojpuri uses Devanagari script; IPA provided for TTS in Part 3

HINGLISH_BHOJPURI_CORPUS = {

    # ── Physics / Quantum Mechanics (HC Verma lecture) ───────
    "quantum":          ("क्वांटम",         "kʷɑːntəm"),
    "physics":          ("भौतिकी",           "bʱɔːtɪkiː"),
    "energy":           ("ऊर्जा",            "uːrd͡ʒɑː"),
    "particle":         ("कण",              "kəɳ"),
    "wave":             ("लहर",             "ləɦər"),
    "frequency":        ("आवृत्ति",           "ɑːʋr̩ttɪ"),
    "wavelength":       ("तरंगदैर्ध्य",        "tərəŋgdɛːrd͡ʱjə"),
    "amplitude":        ("आयाम",            "ɑːjɑːm"),
    "velocity":         ("वेग",             "ʋeːg"),
    "acceleration":     ("त्वरण",            "tʋərəɳ"),
    "force":            ("बल",              "bəl"),
    "mass":             ("द्रव्यमान",          "drəʋjəmɑːn"),
    "momentum":         ("संवेग",            "səŋʋeːg"),
    "potential":        ("विभव",            "ʋɪbʱəʋ"),
    "electric":         ("बिजुरी",           "bɪd͡ʒuriː"),
    "magnetic":         ("चुम्बकीय",          "t͡ʃumbəkiːj"),
    "photon":           ("फोटान",           "pʰoːʈɑːn"),
    "electron":         ("इलेक्ट्रॉन",         "ɪleːkʈrɔːn"),
    "proton":           ("प्रोटोन",           "proːʈoːn"),
    "neutron":          ("न्यूट्रॉन",           "njuːʈrɔːn"),
    "nucleus":          ("नाभिक",           "nɑːbʱɪk"),
    "atom":             ("परमाणु",           "pərəmɑːɳu"),
    "experiment":       ("परीक्षा",           "pəriːkʃɑː"),
    "observation":      ("निरीक्षण",           "nɪriːkʃəɳ"),
    "result":           ("नतीजा",           "nətiːd͡ʒɑː"),
    "identical":        ("एकसमान",          "eːkəsəmɑːn"),
    "probability":      ("संभावना",           "səmbʱɑːʋənɑː"),
    "uncertainty":      ("अनिश्चितता",         "ənɪʃt͡ʃɪttɑː"),
    "superposition":    ("अध्यारोपण",         "əd͡ʱjɑːroːpəɳ"),
    "interference":     ("व्यतिकरण",          "ʋjətɪkərəɳ"),
    "diffraction":      ("विवर्तन",           "ʋɪʋərtən"),
    "refraction":       ("अपवर्तन",          "əpəʋərtən"),
    "reflection":       ("परावर्तन",          "pərɑːʋərtən"),
    "spectrum":         ("वर्णक्रम",           "ʋərɳəkrəm"),
    "light":            ("परकाश",           "pərkɑːʃ"),
    "time":             ("समय",             "səməj"),
    "space":            ("अकाश",            "ɑːkɑːʃ"),
    "interval":         ("अंतराल",           "əntərɑːl"),
    "classical":        ("परंपरागत",          "pərəmpərɑːgət"),
    "modern":           ("आधुनिक",           "ɑːd͡ʱunɪk"),
    "theory":           ("सिद्धांत",           "sɪd͡ːɦɑːnt"),
    "law":              ("नियम",            "nɪjəm"),
    "equation":         ("समीकरण",          "səmiːkərəɳ"),
    "constant":         ("स्थिरांक",           "stʰɪrɑːŋk"),
    "variable":         ("चर",              "t͡ʃər"),
    "function":         ("फलन",            "pʰələn"),
    "derivative":       ("अवकलज",          "əʋəkələd͡ʒ"),
    "integral":         ("समाकल",           "səmɑːkəl"),

    # ── Speech Processing (course content) ───────────────────
    "speech":           ("बोली",            "boːliː"),
    "signal":           ("संकेत",           "səŋkeːt"),
    "noise":            ("शोर",             "ʃoːr"),
    "filter":           ("छनना",            "t͡ʃʰənnɑː"),
    "sampling":         ("नमूना लेवल",        "nəmuːnɑː leːʋəl"),
    "frequency_band":   ("आवृत्ति पट्टी",      "ɑːʋr̩ttɪ pəʈʈiː"),
    "cepstrum":         ("सेप्स्ट्रम",         "seːpstrəm"),
    "cepstral":         ("सेप्स्ट्रल",          "seːpstrəl"),
    "mfcc":             ("एमएफसीसी",        "eːm eːf siː siː"),
    "spectrogram":      ("स्पेक्ट्रोग्राम",      "speːkʈrogrɑːm"),
    "mel":              ("मेल",             "meːl"),
    "filterbank":       ("फिल्टर बैंक",       "pʰɪlʈər bɛːŋk"),
    "phoneme":          ("ध्वनिम",           "d͡ʱʋənɪm"),
    "formant":          ("फॉर्मेंट",           "fɔːrmeːnʈ"),
    "pitch":            ("स्वर ऊंचाई",        "sʋər uːŋt͡ʃɑːiː"),
    "fundamental":      ("मौलिक",           "mɔːlɪk"),
    "harmonic":         ("हारमोनिक",         "ɦɑːrmoːnɪk"),
    "prosody":          ("स्वराघात",          "sʋərɑːgʱɑːt"),
    "intonation":       ("स्वरोच्चारण",        "sʋərot͡ʃt͡ʃɑːrəɳ"),
    "duration":         ("अवधि",            "əʋəd͡ʱɪ"),
    "acoustic":         ("ध्वनिक",           "d͡ʱʋənɪk"),
    "recognition":      ("पहचान",           "pəɦt͡ʃɑːn"),
    "synthesis":        ("संश्लेषण",          "səŋʃleːʃəɳ"),
    "transcription":    ("लिप्यंतरण",         "lɪpjəntərəɳ"),
    "language":         ("भाषा",            "bʱɑːʃɑː"),
    "model":            ("मॉडल",            "mɔːɖəl"),
    "training":         ("सिखावल",          "sɪkʱɑːʋəl"),
    "feature":          ("लच्छन",           "lət͡ʃʃən"),
    "extraction":       ("निकासी",           "nɪkɑːsiː"),
    "classification":   ("वर्गीकरण",          "ʋərgiːkərəɳ"),
    "neural":           ("नस-तंत्र",          "nəs təntrə"),
    "network":          ("जाल",             "d͡ʒɑːl"),
    "deep":             ("गहिर",            "gəɦɪr"),
    "learning":         ("सीखल",           "siːkʰəl"),
    "transformer":      ("ट्रांसफार्मर",        "ʈrɑːnsfɑːrmər"),
    "attention":        ("ध्यान",            "d͡ʱjɑːn"),
    "encoder":          ("एन्कोडर",          "eːnkoːɖər"),
    "decoder":          ("डिकोडर",          "ɖɪkoːɖər"),
    "embedding":        ("अंतःस्थापन",        "əntəstʰɑːpən"),
    "hidden":           ("लुकाइल",          "lʊkɑːɪl"),
    "markov":           ("मार्कोव",           "mɑːrkoːʋ"),
    "gaussian":         ("गाउसियन",          "gɑːusiːən"),
    "viterbi":          ("विटर्बी",           "ʋɪʈərbiː"),
    "beam":             ("किरण",            "kɪrəɳ"),
    "decoding":         ("डिकोडिंग",         "ɖɪkoːɖɪŋ"),
    "alignment":        ("संरेखण",           "səŋreːkʰəɳ"),
    "corpus":           ("भाषा संग्रह",       "bʱɑːʃɑː səŋgrəɦ"),
    "vocabulary":       ("शब्द भंडार",        "ʃəbd bʱənɖɑːr"),
    "token":            ("टोकन",            "ʈoːkən"),
    "perplexity":       ("उलझन माप",        "ʊld͡ʒʱən mɑːp"),
    "whisper":          ("फुसफुसावल",        "pʰʊspʰʊsɑːʋəl"),
    "microphone":       ("माइक",            "mɑːɪk"),
    "recording":        ("रिकार्डिंग",         "rɪkɑːrɖɪŋ"),
    "speaker":          ("बोलेवाला",         "boːleːʋɑːlɑː"),
    "voice":            ("आवाज",            "ɑːʋɑːd͡ʒ"),
    "cloning":          ("नकल बनावल",       "nəkəl bənɑːʋəl"),
    "synthesis_tts":    ("आवाज बनावल",      "ɑːʋɑːd͡ʒ bənɑːʋəl"),
    "waveform":         ("तरंग रूप",         "tərəŋg ruːp"),
    "spectrogram_mel":  ("मेल चित्र",        "meːl t͡ʃɪtr"),
    "vocoder":          ("वोकोडर",          "ʋoːkoːɖər"),
    "dtw":              ("गतिशील समय वार्पिंग", "gətɪʃiːl səməj ʋɑːrpɪŋ"),
    "warping":          ("मोड़ल",            "moːɽəl"),
    "prosody_transfer": ("स्वर शैली बदलावल",  "sʋər ʃɛːliː bədəlɑːʋəl"),
    "zero_shot":        ("बिना सिखले",       "bɪnɑː sɪkʱle"),
    "low_resource":     ("कम संसाधन",       "kəm sənsɑːd͡ʱən"),
    "code_switch":      ("भाषा मिलावल",      "bʱɑːʃɑː mɪlɑːʋəl"),
    "hinglish":         ("हिंग्लिश",          "ɦɪŋglɪʃ"),
    "bhojpuri":         ("भोजपुरी",          "bʱoːd͡ʒpuriː"),
    "devanagari":       ("देवनागरी",          "deːʋənɑːgəriː"),
    "ipa":              ("आईपीए",           "ɑːiːpiːeː"),
    "phonetic":         ("ध्वन्यात्मक",        "d͡ʱʋənjɑːtmək"),
    "grapheme":         ("लिपि-चिन्ह",       "lɪpɪ t͡ʃɪnʱ"),
    "denoising":        ("शोर हटावल",        "ʃoːr ɦəʈɑːʋəl"),
    "reverb":           ("गूंज",             "guːnd͡ʒ"),
    "background":       ("पृष्ठभूमि",          "pr̩ʃʈʰbʱuːmɪ"),

    # ── Common Hinglish connectives (lecture language) ────────
    "तो":              ("त",               "tə"),
    "और":              ("अउर",             "əʊr"),
    "है":              ("बा",               "bɑː"),
    "हैं":             ("बाड़न",            "bɑːɽən"),
    "के":              ("के",               "keː"),
    "का":              ("के",               "keː"),
    "की":              ("के",               "keː"),
    "में":             ("में",              "meː"),
    "से":              ("से",               "seː"),
    "पर":              ("पर",              "pər"),
    "यह":              ("इ",               "ɪ"),
    "वह":              ("ऊ",               "uː"),
    "हम":              ("हम",              "ɦəm"),
    "आप":              ("रउआ",            "rəʊɑː"),
    "क्या":            ("का",              "kɑː"),
    "कैसे":            ("केने",            "keːne"),
    "क्यों":           ("काहे",            "kɑːɦe"),
    "जो":              ("जे",              "d͡ʒeː"),
    "कि":              ("कि",              "kɪ"),
    "लेकिन":           ("बाकी",            "bɑːkiː"),
    "अगर":             ("अगर",            "əgər"),
    "तब":              ("तब",             "təb"),
    "जब":              ("जब",             "d͡ʒəb"),
    "सब":              ("सब",             "səb"),
    "कुछ":             ("कुछ",            "kʊt͡ʃʰ"),
    "बहुत":            ("बहुते",           "bəɦuteː"),
    "बड़ा":            ("बड़",             "bəɽ"),
    "छोटा":            ("छोट",            "t͡ʃʰoːʈ"),
    "अच्छा":           ("नीमन",           "niːmən"),
    "हाँ":             ("हँ",              "ɦə̃"),
    "नहीं":            ("ना",             "nɑː"),
    "देखना":           ("देखल",           "deːkʰəl"),
    "करना":            ("करल",           "kərəl"),
    "होना":            ("होखल",           "ɦoːkʰəl"),
    "समझना":           ("बुझल",           "bʊd͡ʒʰəl"),
    "बताना":           ("बतावल",          "bətɑːʋəl"),
    "जानना":           ("जानल",          "d͡ʒɑːnəl"),
    "पढ़ना":           ("पढ़ल",           "pəɽʰəl"),
    "लिखना":           ("लिखल",          "lɪkʰəl"),

    # ── Numbers and math ─────────────────────────────────────
    "zero":             ("सून",            "suːn"),
    "one":              ("एक",             "eːk"),
    "two":              ("दू",             "duː"),
    "three":            ("तीन",            "tiːn"),
    "four":             ("चार",            "t͡ʃɑːr"),
    "five":             ("पांच",           "pɑːnt͡ʃ"),
    "equal":            ("बराबर",          "bərɑːbər"),
    "plus":             ("जोड़",            "d͡ʒoːɽ"),
    "minus":            ("घटावल",          "gʰəʈɑːʋəl"),
    "multiply":         ("गुना",            "gunɑː"),
    "divide":           ("भाग",            "bʱɑːg"),
    "matrix":           ("आव्यूह",          "ɑːʋjuːɦ"),
    "vector":           ("सदिश",           "sədɪʃ"),
    "scalar":           ("अदिश",           "ədɪʃ"),
    "dimension":        ("आयाम",           "ɑːjɑːm"),
    "coordinate":       ("निर्देशांक",        "nɪrdeːʃɑːŋk"),
}


# ─────────────────────────────────────────────────────────────
# TASK 2.1 — HINGLISH G2P MAPPING LAYER
# ─────────────────────────────────────────────────────────────

# IPA mappings for Hindi consonants (Devanagari → IPA)
HINDI_CONSONANT_MAP = {
    "क": "k",   "ख": "kʰ",  "ग": "g",   "घ": "gʱ",  "ङ": "ŋ",
    "च": "t͡ʃ", "छ": "t͡ʃʰ","ज": "d͡ʒ", "झ": "d͡ʒʱ","ञ": "ɲ",
    "ट": "ʈ",   "ठ": "ʈʰ",  "ड": "ɖ",   "ढ": "ɖʱ",  "ण": "ɳ",
    "त": "t",   "थ": "tʰ",  "द": "d",   "ध": "dʱ",  "न": "n",
    "प": "p",   "फ": "pʰ",  "ब": "b",   "भ": "bʱ",  "म": "m",
    "य": "j",   "र": "r",   "ल": "l",   "व": "ʋ",
    "श": "ʃ",   "ष": "ʂ",   "स": "s",   "ह": "ɦ",
    "क्ष": "kʂ","त्र": "tr", "ज्ञ": "gj",
    "ड़": "ɽ",  "ढ़": "ɽʱ",
    "ं": "̃",   "ः": "ɦ",   "ँ": "̃",
}

# IPA mappings for Hindi vowels
HINDI_VOWEL_MAP = {
    "अ": "ə",  "आ": "ɑː", "इ": "ɪ",  "ई": "iː",
    "उ": "ʊ",  "ऊ": "uː", "ए": "eː", "ऐ": "ɛː",
    "ओ": "oː", "औ": "ɔː", "ऋ": "r̩",
    "ा": "ɑː", "ि": "ɪ",  "ी": "iː", "ु": "ʊ",
    "ू": "uː", "े": "eː", "ै": "ɛː", "ो": "oː",
    "ौ": "ɔː", "ृ": "r̩",  "्": "",   "़": "",
}

# English to IPA (CMU-style simplified mapping for common patterns)
ENGLISH_G2P_RULES = [
    # Digraphs first (order matters)
    (r"ph",   "f"),
    (r"th",   "θ"),
    (r"sh",   "ʃ"),
    (r"ch",   "t͡ʃ"),
    (r"gh",   "g"),
    (r"ck",   "k"),
    (r"ng",   "ŋ"),
    (r"wh",   "w"),
    (r"tion", "ʃən"),
    (r"sion", "ʒən"),
    (r"ture", "t͡ʃər"),
    (r"ous",  "əs"),
    (r"ious", "iːəs"),
    (r"ful",  "fʊl"),
    (r"less", "ləs"),
    (r"ness", "nəs"),
    (r"ment", "mənt"),
    (r"tion", "ʃən"),
    (r"ing",  "ɪŋ"),
    (r"ed$",  "d"),
    (r"er$",  "ər"),
    (r"est$", "ɪst"),
    (r"ly$",  "liː"),
    # Vowel patterns
    (r"ee",   "iː"),
    (r"ea",   "iː"),
    (r"oo",   "uː"),
    (r"ou",   "aʊ"),
    (r"ow",   "oʊ"),
    (r"oi",   "ɔɪ"),
    (r"ai",   "eɪ"),
    (r"ay",   "eɪ"),
    (r"ie",   "iː"),
    (r"a",    "æ"),
    (r"e",    "ɛ"),
    (r"i",    "ɪ"),
    (r"o",    "ɒ"),
    (r"u",    "ʌ"),
    # Consonants
    (r"c(?=[ei])", "s"),
    (r"c",    "k"),
    (r"g(?=[ei])", "d͡ʒ"),
    (r"g",    "g"),
    (r"j",    "d͡ʒ"),
    (r"q",    "k"),
    (r"x",    "ks"),
    (r"y(?=[aeiou])", "j"),
    (r"y$",   "iː"),
    (r"z",    "z"),
    (r"w",    "w"),
    (r"r",    "r"),
    (r"l",    "l"),
    (r"m",    "m"),
    (r"n",    "n"),
    (r"p",    "p"),
    (r"b",    "b"),
    (r"t",    "t"),
    (r"d",    "d"),
    (r"f",    "f"),
    (r"v",    "v"),
    (r"k",    "k"),
    (r"s",    "s"),
    (r"h",    "h"),
]

# Bhojpuri-specific phonological adjustments from Hindi IPA
# Some Hindi phonemes shift in Bhojpuri
HINDI_TO_BHOJPURI_IPA = {
    # Verb endings: Hindi -nā → Bhojpuri -al/-l
    "nɑː":  "əl",
    # है → बा (copula shift)
    "ɦɛː":  "bɑː",
    "hɛː":  "bɑː",
    # को → के (dative marker)
    "koː":  "keː",
    # Retroflex flap more prominent
    "ɽ":    "ɽ",
}


class WordLanguageDetector:
    """
    Word-level language detector for Hinglish text.
    Routes each word to the correct G2P engine.

    Detection strategy:
      1. If word contains Devanagari Unicode chars → Hindi
      2. If word is in our corpus → use corpus language tag
      3. If word matches common English patterns → English
      4. Default → English (Roman script = likely English/loanword)
    """

    # Unicode ranges
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    DEVANAGARI_EXT   = (0xA8E0, 0xA8FF)

    # Common Hindi words written in Roman script (code-switching)
    ROMAN_HINDI_WORDS = {
        "hai", "hain", "ka", "ke", "ki", "ko", "se", "par", "mein",
        "aur", "ya", "to", "jo", "kya", "kaise", "kyun", "kyunki",
        "lekin", "agar", "tab", "jab", "sab", "kuch", "bahut",
        "nahi", "haan", "theek", "accha", "matlab", "yani",
        "matlab", "dekho", "suno", "samjhe", "batao", "karo",
        "hota", "hoti", "hote", "wala", "wali", "wale",
        "phir", "abhi", "pehle", "baad", "saath", "bina",
        "iske", "uske", "humara", "tumhara", "apna", "apni",
        "ek", "do", "teen", "char", "paanch",
    }

    def __init__(self, corpus: Dict):
        self.corpus      = corpus
        self.corpus_keys = set(corpus.keys())

    def is_devanagari(self, word: str) -> bool:
        """Check if word contains Devanagari script."""
        for ch in word:
            cp = ord(ch)
            if (self.DEVANAGARI_RANGE[0] <= cp <= self.DEVANAGARI_RANGE[1] or
                    self.DEVANAGARI_EXT[0] <= cp <= self.DEVANAGARI_EXT[1]):
                return True
        return False

    def detect(self, word: str) -> str:
        """
        Returns: "hindi" | "english" | "mixed"
        """
        clean = word.lower().strip(".,!?;:\"'()-")

        if not clean:
            return "english"

        # Devanagari script → definitely Hindi
        if self.is_devanagari(clean):
            return "hindi"

        # Known Roman Hindi words
        if clean in self.ROMAN_HINDI_WORDS:
            return "hindi"

        # In corpus as English technical term
        if clean in self.corpus_keys:
            return "english"

        # All Roman script → treat as English
        if all(ord(c) < 128 for c in clean):
            return "english"

        return "english"


class HindiG2P:
    """
    Grapheme-to-Phoneme for Hindi (Devanagari script).
    Rule-based mapping using the Devanagari Unicode block.

    Handles:
      - Vowel diacritics (matras)
      - Consonant clusters with halant (्)
      - Nasalization (anusvara ं, chandrabindu ँ)
      - Visarga (ः)
      - Nukta characters (ड़, ढ़, etc.)
    """

    def __init__(self):
        self.consonants = HINDI_CONSONANT_MAP
        self.vowels     = HINDI_VOWEL_MAP

    def convert(self, word: str) -> str:
        """Convert a Hindi (Devanagari) word to IPA."""
        ipa    = []
        i      = 0
        chars  = list(word)

        while i < len(chars):
            ch = chars[i]

            # Check for two-char sequences (conjuncts defined in map)
            if i + 1 < len(chars):
                bigram = ch + chars[i+1]
                if bigram in self.consonants:
                    ipa.append(self.consonants[bigram])
                    i += 2
                    continue

            # Single character lookup
            if ch in self.consonants:
                ipa.append(self.consonants[ch])
                # Add inherent vowel 'ə' unless followed by halant or matra
                next_ch = chars[i+1] if i+1 < len(chars) else ""
                if next_ch != "्" and next_ch not in self.vowels:
                    ipa.append("ə")
            elif ch in self.vowels:
                v = self.vowels[ch]
                if v:  # skip empty (halant)
                    ipa.append(v)
            else:
                # Unknown character — keep as-is (punctuation, etc.)
                pass

            i += 1

        result = "".join(ipa)
        # Clean up: remove double schwa, fix common patterns
        result = result.replace("əə", "ə")
        result = result.replace("ɑːə", "ɑː")
        return result


class EnglishG2P:
    """
    Rule-based English G2P for technical speech/physics terms.
    Uses ordered regex substitution rules.

    For production: replace with epitran or g2p_en library.
    This rule-based version handles the specific vocabulary
    in the HC Verma lectures adequately.
    """

    # Pre-built dictionary for common technical terms
    # (overrides rules for irregular pronunciations)
    EXCEPTIONS = {
        "cepstrum":      "sɛpstrəm",
        "cepstral":      "sɛpstrəl",
        "stochastic":    "stəkæstɪk",
        "gaussian":      "gaʊsiːən",
        "viterbi":       "vɪtərbiː",
        "mel":           "mɛl",
        "mfcc":          "ɛm ɛf siː siː",
        "hmm":           "eɪt͡ʃ ɛm ɛm",
        "lstm":          "ɛl ɛs tiː ɛm",
        "ctc":           "siː tiː siː",
        "dtw":           "diː tiː dʌbljuː",
        "ipa":           "aɪ piː eɪ",
        "wav2vec":       "wɛv tuː vɛk",
        "whisper":       "wɪspər",
        "quantum":       "kwɒntəm",
        "photon":        "foʊtɒn",
        "electron":      "ɪlɛktrɒn",
        "nucleus":       "njuːkliːəs",
        "frequency":     "friːkwənsiː",
        "wavelength":    "weɪvlɛŋθ",
        "amplitude":     "æmplɪtjuːd",
        "velocity":      "vəlɒsɪtiː",
        "momentum":      "məmɛntəm",
        "interference":  "ɪntərfɪərəns",
        "diffraction":   "dɪfrækʃən",
        "superposition": "suːpərpəzɪʃən",
        "uncertainty":   "ʌnsɜːtəntiː",
        "identical":     "aɪdɛntɪkəl",
        "probability":   "prɒbəbɪlɪtiː",
        "phoneme":       "foʊniːm",
        "prosody":       "prɒsədiː",
        "formant":       "fɔːrmænt",
        "spectrogram":   "spɛktrəgræm",
        "vocoder":       "voʊkoʊdər",
        "acoustic":      "əkuːstɪk",
        "perplexity":    "pərplɛksɪtiː",
        "transformer":   "trænsfoːrmər",
        "attention":     "ətɛnʃən",
    }

    def __init__(self):
        # Compile regex rules once
        self.rules = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in ENGLISH_G2P_RULES
        ]

    def convert(self, word: str) -> str:
        """Convert an English word to approximate IPA."""
        lower = word.lower().strip(".,!?;:\"'()-")

        # Check exceptions dictionary first
        if lower in self.EXCEPTIONS:
            return self.EXCEPTIONS[lower]

        # Apply rules sequentially
        result = lower
        for pattern, replacement in self.rules:
            result = pattern.sub(replacement, result)

        return result


class HinglishG2P:
    """
    Unified G2P for Hinglish (code-switched Hindi+English) text.

    Pipeline per word:
        1. Detect language (Hindi / English)
        2. Route to HindiG2P or EnglishG2P
        3. Look up corpus for technical terms (highest priority)
        4. Apply Bhojpuri phonological adjustments for translation

    This is the "custom mapping layer for Hinglish phonology"
    required by Task 2.1.
    """

    def __init__(self, corpus: Dict = None):
        self.corpus   = corpus or HINGLISH_BHOJPURI_CORPUS
        self.detector = WordLanguageDetector(self.corpus)
        self.hindi_g2p   = HindiG2P()
        self.english_g2p = EnglishG2P()

    def word_to_ipa(self, word: str, language_hint: str = None) -> Tuple[str, str, str]:
        """
        Convert a single word to IPA.

        Returns:
            (original_word, detected_language, ipa_string)
        """
        clean = word.strip(".,!?;:\"'()-\n ")
        if not clean:
            return (word, "punct", "")

        # Corpus lookup (highest priority — handles technical terms)
        clean_lower = clean.lower()
        if clean_lower in self.corpus:
            bhojpuri_word, ipa = self.corpus[clean_lower]
            return (word, "corpus", ipa)

        # Language detection
        lang = language_hint or self.detector.detect(clean)

        if lang == "hindi":
            ipa = self.hindi_g2p.convert(clean)
        else:
            ipa = self.english_g2p.convert(clean)

        return (word, lang, ipa)

    def segment_to_ipa(
        self,
        text:          str,
        language_hint: str = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Convert a full segment of text to IPA, word by word.

        Returns:
            List of (word, language, ipa) tuples
        """
        # Tokenize preserving punctuation as separate tokens
        tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+|[0-9]+|[^\w\s]|\s+", text)

        results = []
        for token in tokens:
            if token.isspace() or not token.strip():
                results.append((token, "space", " "))
                continue
            if re.match(r"[^\w\u0900-\u097F]", token):
                results.append((token, "punct", token))
                continue

            word, lang, ipa = self.word_to_ipa(token, language_hint)
            results.append((word, lang, ipa))

        return results

    def text_to_ipa_string(
        self,
        text:          str,
        language_hint: str = None,
        separator:     str = " ",
    ) -> str:
        """
        Convert full text to a single IPA string.
        """
        word_ipas = self.segment_to_ipa(text, language_hint)
        ipa_parts = [ipa for _, lang, ipa in word_ipas
                     if lang not in ("space", "punct") and ipa]
        return separator.join(ipa_parts)


# ─────────────────────────────────────────────────────────────
# TASK 2.2 — BHOJPURI TRANSLATOR
# ─────────────────────────────────────────────────────────────

class BhojpuriTranslator:
    """
    Translates Hinglish (Hindi+English) text into Bhojpuri.

    Strategy (in priority order):
      1. Exact corpus lookup for known technical terms
      2. Rule-based Hindi→Bhojpuri morphological transformation
      3. English loanwords kept as-is (Bhojpuri borrows freely)
      4. Unknown words → transliterated with Bhojpuri phonology

    Key Hindi→Bhojpuri grammatical rules implemented:
      - Verb infinitive: -ना (-nā) → -ल (-l) / -ब (-b) [future]
      - Copula: है → बा, हैं → बाड़न
      - Perfective: -या (-yā) → -ल (-l)
      - Postposition: को → के, में → में (same), से → से (same)
      - Pronoun: वह → ऊ, आप → रउआ, तुम → तू
      - Plural: -एं → -अन
    """

    def __init__(self, corpus: Dict = None):
        self.corpus   = corpus or HINGLISH_BHOJPURI_CORPUS
        self.detector = WordLanguageDetector(self.corpus)

        # Morphological substitution rules: (pattern, replacement)
        # Applied to Hindi Devanagari text
        self.morph_rules = [
            # Copula
            (r"है\b",      "बा"),
            (r"हैं\b",     "बाड़न"),
            (r"था\b",      "रहल"),
            (r"थे\b",      "रहलन"),
            (r"थी\b",      "रहल"),
            (r"होगा\b",    "होई"),
            (r"होगी\b",    "होई"),
            (r"होंगे\b",   "होईं"),

            # Pronouns
            (r"\bवह\b",    "ऊ"),
            (r"\bवो\b",    "ऊ"),
            (r"\bआप\b",    "रउआ"),
            (r"\bतुम\b",   "तू"),
            (r"\bहम\b",    "हम"),
            (r"\bमैं\b",   "हम"),

            # Postpositions
            (r"\bको\b",    "के"),
            (r"\bका\b",    "के"),
            (r"\bकी\b",    "के"),

            # Verb endings
            (r"ना\b",      "ल"),
            (r"ते हैं\b",  "तानी"),
            (r"ती है\b",   "तिया"),
            (r"ता है\b",   "तारे"),
            (r"रहा है\b",  "बा"),
            (r"रही है\b",  "बिया"),
            (r"कर\b",      "करके"),

            # Common words
            (r"\bअच्छा\b", "नीमन"),
            (r"\bबहुत\b",  "बहुते"),
            (r"\bबड़ा\b",  "बड़"),
            (r"\bछोटा\b",  "छोट"),
            (r"\bनहीं\b",  "ना"),
            (r"\bक्यों\b", "काहे"),
            (r"\bकैसे\b",  "केने"),
            (r"\bक्या\b",  "का"),
        ]
        self.compiled_rules = [
            (re.compile(p), r) for p, r in self.morph_rules
        ]

    def translate_word(self, word: str) -> Tuple[str, str]:
        """
        Translate a single word to Bhojpuri.

        Returns:
            (bhojpuri_word, translation_method)
        """
        clean = word.strip(".,!?;:\"'()-\n ")
        clean_lower = clean.lower()

        # Corpus lookup
        if clean_lower in self.corpus:
            bhojpuri, _ = self.corpus[clean_lower]
            return (bhojpuri, "corpus")

        # Devanagari → apply morphological rules
        if self.detector.is_devanagari(clean):
            result = clean
            for pattern, replacement in self.compiled_rules:
                result = pattern.sub(replacement, result)
            method = "morphological" if result != clean else "unchanged"
            return (result, method)

        # English technical terms → keep as-is (Bhojpuri borrows English)
        if all(ord(c) < 128 for c in clean):
            return (clean, "loanword")

        return (word, "unchanged")

    def translate_segment(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Translate a full segment to Bhojpuri.

        Returns:
            (bhojpuri_text, word_alignment_list)
        """
        tokens    = re.findall(
            r"[\u0900-\u097F]+|[a-zA-Z]+|[0-9]+|[^\w\s]|\s+", text
        )
        output_tokens  = []
        alignment      = []

        for token in tokens:
            if token.isspace():
                output_tokens.append(token)
                continue
            if not token.strip():
                continue
            if re.match(r"^[^\w\u0900-\u097F]+$", token):
                output_tokens.append(token)
                continue

            bhojpuri, method = self.translate_word(token)
            output_tokens.append(bhojpuri)
            alignment.append({
                "source": token,
                "target": bhojpuri,
                "method": method,
            })

        return ("".join(output_tokens), alignment)

    def translate_full(
        self,
        transcript: Dict,
        output_json: str,
        output_txt:  str,
    ) -> Dict:
        """
        Translate all segments in a Whisper transcript to Bhojpuri.
        """
        translated_segments = []

        for seg in transcript.get("segments", []):
            original_text = seg.get("text", "").strip()
            dom_lang      = seg.get("dominant_language", "unknown")

            bhojpuri_text, alignment = self.translate_segment(original_text)

            translated_segments.append({
                "start":           seg.get("start"),
                "end":             seg.get("end"),
                "original":        original_text,
                "bhojpuri":        bhojpuri_text,
                "dominant_language": dom_lang,
                "alignment":       alignment,
                "n_corpus_hits":   sum(1 for a in alignment if a["method"] == "corpus"),
                "n_morph_rules":   sum(1 for a in alignment if a["method"] == "morphological"),
                "n_loanwords":     sum(1 for a in alignment if a["method"] == "loanword"),
            })

        result = {
            "language_pair":     "hinglish → bhojpuri",
            "total_segments":    len(translated_segments),
            "segments":          translated_segments,
        }

        # Save JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save readable TXT
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("# Hinglish → Bhojpuri Translation\n")
            f.write("# Format: [time | LANG] Original || Bhojpuri\n\n")
            for seg in translated_segments:
                start = seg["start"]
                end   = seg["end"]
                lang  = seg["dominant_language"].upper()
                f.write(
                    f"[{start:.2f}s → {end:.2f}s | {lang}]\n"
                    f"  HI/EN : {seg['original']}\n"
                    f"  BHOJ  : {seg['bhojpuri']}\n\n"
                )

        logger.info(f"Bhojpuri translation saved: {output_json}, {output_txt}")
        n_corpus = sum(s["n_corpus_hits"] for s in translated_segments)
        n_morph  = sum(s["n_morph_rules"] for s in translated_segments)
        logger.info(f"Translation stats — Corpus hits: {n_corpus}, "
                    f"Morphological: {n_morph}")

        return result


# ─────────────────────────────────────────────────────────────
# FULL IPA TRANSCRIPTION PIPELINE
# ─────────────────────────────────────────────────────────────

def run_ipa_conversion(
    transcript:     Dict,
    g2p:            HinglishG2P,
    output_json:    str,
    output_txt:     str,
) -> Dict:
    """
    Convert all segments in transcript to IPA strings.
    Produces unified IPA representation for the full lecture.
    """
    ipa_segments = []

    for seg in transcript.get("segments", []):
        text      = seg.get("text", "").strip()
        dom_lang  = seg.get("dominant_language", "unknown")

        # Get word-level IPA
        word_ipas = g2p.segment_to_ipa(text, dom_lang)

        # Build full IPA string
        ipa_words = [
            ipa for _, lang, ipa in word_ipas
            if lang not in ("space", "punct") and ipa
        ]
        ipa_string = " ".join(ipa_words)

        # Language breakdown
        lang_breakdown = defaultdict(int)
        for _, lang, _ in word_ipas:
            if lang not in ("space", "punct"):
                lang_breakdown[lang] += 1

        ipa_segments.append({
            "start":          seg.get("start"),
            "end":            seg.get("end"),
            "original_text":  text,
            "ipa_string":     ipa_string,
            "word_ipa":       [
                {"word": w, "language": l, "ipa": i}
                for w, l, i in word_ipas
                if l not in ("space",) and w.strip()
            ],
            "dominant_language": dom_lang,
            "language_breakdown": dict(lang_breakdown),
        })

    result = {
        "total_segments": len(ipa_segments),
        "g2p_method":     "hinglish_custom_layer",
        "segments":       ipa_segments,
    }

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save readable TXT
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("# Hinglish IPA Transcript\n")
        f.write("# Format: [time | LANG] Text → IPA\n\n")
        for seg in ipa_segments:
            f.write(
                f"[{seg['start']:.2f}s → {seg['end']:.2f}s | "
                f"{seg['dominant_language'].upper()}]\n"
                f"  TEXT : {seg['original_text']}\n"
                f"  IPA  : /{seg['ipa_string']}/\n\n"
            )

    logger.info(f"IPA transcript saved: {output_json}, {output_txt}")
    return result


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def evaluate_part2_metrics(
    ipa_result:         Dict,
    translation_result: Dict,
    output_json:        str,
) -> Dict:
    """
    Evaluate Part 2 quality metrics for the report.

    Metrics computed:
      - Corpus coverage: % of words covered by corpus
      - G2P language distribution: English/Hindi/corpus word counts
      - Translation method breakdown
      - Corpus size validation (must be ≥500 words)
    """
    # Corpus coverage
    total_words   = 0
    corpus_hits   = 0
    en_words      = 0
    hi_words      = 0

    for seg in ipa_result.get("segments", []):
        for w in seg.get("word_ipa", []):
            total_words += 1
            lang = w["language"]
            if lang == "corpus":
                corpus_hits += 1
            elif lang == "english":
                en_words += 1
            elif lang == "hindi":
                hi_words += 1

    coverage = corpus_hits / max(total_words, 1)

    # Translation breakdown
    total_translated = 0
    by_method        = defaultdict(int)
    for seg in translation_result.get("segments", []):
        for a in seg.get("alignment", []):
            by_method[a["method"]] += 1
            total_translated += 1

    # Corpus size
    corpus_size = len(HINGLISH_BHOJPURI_CORPUS)

    metrics = {
        "corpus_stats": {
            "total_entries":    corpus_size,
            "passes_500_req":   corpus_size >= 500,
            "threshold":        500,
        },
        "g2p_stats": {
            "total_words":      total_words,
            "corpus_hits":      corpus_hits,
            "english_words":    en_words,
            "hindi_words":      hi_words,
            "corpus_coverage":  round(coverage, 4),
        },
        "translation_stats": {
            "total_tokens":     total_translated,
            "by_method":        dict(by_method),
        },
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("PART 2 METRICS")
    logger.info("=" * 60)
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    logger.info(f"  Corpus size    : {corpus_size:4d} entries  "
                f"{'(≥500 required)':20s}  "
                f"{PASS if corpus_size >= 500 else FAIL}")
    logger.info(f"  G2P coverage   : {coverage:.1%} words from corpus")
    logger.info(f"  English words  : {en_words}")
    logger.info(f"  Hindi words    : {hi_words}")
    logger.info(f"  Corpus hits    : {corpus_hits}")
    logger.info(f"  Translation    : {dict(by_method)}")
    logger.info("=" * 60)

    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Part 2 metrics saved: {output_json}")

    return metrics


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_part2(
    transcript_path: str,
    mode:            str = "full",
):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("corpus",  exist_ok=True)

    ipa_json         = "outputs/ipa_transcript.json"
    ipa_txt          = "outputs/ipa_transcript.txt"
    bhoj_json        = "outputs/bhojpuri_translation.json"
    bhoj_txt         = "outputs/bhojpuri_translation.txt"
    metrics_json     = "outputs/part2_metrics.json"
    corpus_json      = "corpus/hinglish_bhojpuri_corpus.json"

    # Save corpus to file for inspection / report
    with open(corpus_json, "w", encoding="utf-8") as f:
        json.dump(
            {k: {"bhojpuri": v[0], "ipa": v[1]}
             for k, v in HINGLISH_BHOJPURI_CORPUS.items()},
            f, indent=2, ensure_ascii=False
        )
    logger.info(f"Corpus saved ({len(HINGLISH_BHOJPURI_CORPUS)} entries): {corpus_json}")

    # Load transcript from Part 1
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(
            f"Transcript not found: {transcript_path}\n"
            f"Run Part 1 first: python ../part1/part1_transcription.py "
            f"--audio ../data/lecture_segment.wav --mode transcribe"
        )

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    logger.info(f"Loaded transcript: {len(transcript.get('segments', []))} segments")

    # Initialise G2P and translator
    g2p        = HinglishG2P(HINGLISH_BHOJPURI_CORPUS)
    translator = BhojpuriTranslator(HINGLISH_BHOJPURI_CORPUS)

    ipa_result         = None
    translation_result = None

    # ── Task 2.1: IPA conversion ──────────────────────────────
    if mode in ("full", "ipa"):
        logger.info("Running Task 2.1: Hinglish → IPA conversion...")
        ipa_result = run_ipa_conversion(transcript, g2p, ipa_json, ipa_txt)

    # ── Task 2.2: Bhojpuri translation ───────────────────────
    if mode in ("full", "translate"):
        logger.info("Running Task 2.2: Hinglish → Bhojpuri translation...")
        translation_result = translator.translate_full(transcript, bhoj_json, bhoj_txt)

    # ── Metrics ───────────────────────────────────────────────
    if ipa_result and translation_result:
        evaluate_part2_metrics(ipa_result, translation_result, metrics_json)

    logger.info("=" * 60)
    logger.info("PART 2 COMPLETE")
    logger.info(f"  IPA transcript      : {ipa_txt}")
    logger.info(f"  Bhojpuri translation: {bhoj_txt}")
    logger.info(f"  Corpus              : {corpus_json}")
    logger.info(f"  Metrics             : {metrics_json}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part II: Phonetic Mapping & Bhojpuri Translation"
    )
    parser.add_argument(
        "--transcript", type=str,
        default="../part1/outputs/transcript.json",
        help="Path to Part 1 transcript JSON (default: ../part1/outputs/transcript.json)"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "ipa", "translate"],
        help="Pipeline mode: full | ipa | translate (default: full)"
    )
    args = parser.parse_args()

    run_part2(
        transcript_path = args.transcript,
        mode            = args.mode,
    )
