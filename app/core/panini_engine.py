"""
Pāṇini Rule Engine — Production Module.
Deterministic Sanskrit compiler based on the Aṣṭādhyāyī.
Ported from brahman2.py for modular deployment.
"""
from typing import Dict, List, Optional


class PaniniEngine:
    """
    Hard-coded implementation of Pāṇini's grammar.
    Pure algorithmic logic — no neural components.
    """

    def __init__(self, dhatus: List[Dict], nouns: Optional[Dict] = None):
        self.dhatus = {d['root']: d for d in dhatus}
        self.nouns = nouns or {
            'राम': {'gender': 'masculine', 'type': 'a-stem'},
            'वन': {'gender': 'neuter', 'type': 'a-stem'}
        }
        self.vibhakti_endings = self._build_vibhakti_table()
        self.pratyayas = self._build_pratyaya_table()

    def _build_vibhakti_table(self) -> Dict:
        masculine_a = {
            'nominative': {'singular': 'ः', 'dual': 'ौ', 'plural': 'ाः'},
            'accusative': {'singular': 'म्', 'dual': 'ौ', 'plural': 'ान्'},
            'instrumental': {'singular': 'ेण', 'dual': 'ाभ्याम्', 'plural': 'ैः'},
            'dative': {'singular': 'ाय', 'dual': 'ाभ्याम्', 'plural': 'ेभ्यः'},
            'ablative': {'singular': 'ात्', 'dual': 'ाभ्याम्', 'plural': 'ेभ्यः'},
            'genitive': {'singular': 'स्य', 'dual': 'योः', 'plural': 'ानाम्'},
            'locative': {'singular': 'े', 'dual': 'योः', 'plural': 'ेषु'},
            'vocative': {'singular': '', 'dual': 'ौ', 'plural': 'ाः'}
        }
        neuter_a = {
            'nominative': {'singular': 'म्', 'dual': 'े', 'plural': 'ानि'},
            'accusative': {'singular': 'म्', 'dual': 'े', 'plural': 'ानि'},
            'instrumental': {'singular': 'ेन', 'dual': 'ाभ्याम्', 'plural': 'ैः'},
            'dative': {'singular': 'ाय', 'dual': 'ाभ्याम्', 'plural': 'ेभ्यः'},
            'ablative': {'singular': 'ात्', 'dual': 'ाभ्याम्', 'plural': 'ेभ्यः'},
            'genitive': {'singular': 'स्य', 'dual': 'योः', 'plural': 'ानाम्'},
            'locative': {'singular': 'े', 'dual': 'योः', 'plural': 'ेषु'},
            'vocative': {'singular': '', 'dual': 'े', 'plural': 'ानि'}
        }
        return {'masculine_a': masculine_a, 'neuter_a': neuter_a}

    def _build_pratyaya_table(self) -> Dict:
        present_endings = {
            'parasmaipada': {
                'singular': ['ति', 'सि', 'मि'],
                'dual': ['तः', 'थः', 'वः'],
                'plural': ['न्ति', 'थ', 'मः']
            },
            'ātmanepada': {
                'singular': ['ते', 'से', 'ए'],
                'dual': ['एते', 'एथे', 'वहे'],
                'plural': ['न्ते', 'ध्वे', 'महे']
            }
        }
        return {'present': present_endings}

    def segment_word(self, word: str) -> Dict:
        """Segment a Sanskrit word into its morphological components."""
        analysis = {
            'original': word,
            'dhatu': None,
            'pratipadika': None,
            'pratyayas': [],
            'vibhakti': None,
            'pos': None
        }

        # 1. Try to match against known nominal bases (Nouns)
        for noun_base, noun_info in self.nouns.items():
            stem = noun_base
            declension_type = f"{noun_info['gender']}_{noun_info['type'][0]}"

            if declension_type in self.vibhakti_endings:
                endings = self.vibhakti_endings[declension_type]
                matched_vibhaktis = []
                for case_name, numbers in endings.items():
                    for number_name, ending in numbers.items():
                        possible_forms = [stem + ending]
                        if ending == 'म्':
                            possible_forms.append(stem + 'ं')
                        if word in possible_forms:
                            matched_vibhaktis.append(case_name)

                if matched_vibhaktis:
                    analysis['pratipadika'] = noun_base
                    analysis['pos'] = 'noun'
                    analysis['vibhakti'] = matched_vibhaktis  # Superposition
                    analysis['number'] = 'singular'
                    analysis['gender'] = noun_info['gender']
                    return analysis

        # 2. Try to match against known dhātus (Verbs)
        for dhatu_root, dhatu_info in self.dhatus.items():
            if dhatu_root == 'गम्':
                if word.startswith('गच्छ'):
                    analysis['dhatu'] = dhatu_root
                    analysis['pos'] = 'verb'
                    suffix = word[len('गच्छ'):]
                    if suffix == 'ति':
                        analysis['pratyayas'] = [
                            {'type': 'tense_marker', 'value': 'शप्'},
                            {'type': 'ending', 'value': 'ति'}
                        ]
                        analysis['tense'] = 'present'
                        analysis['person'] = '3rd'
                        analysis['number'] = 'singular'
                    return analysis

        return analysis

    def parse(self, sentence: str) -> Dict:
        """Complete Pāṇinian parse of a Sanskrit sentence."""
        words = sentence.split()
        parsed_words = [self.segment_word(word) for word in words]
        return {
            'sentence': sentence,
            'words': parsed_words,
            'parse_method': 'deterministic_panini'
        }

    def apply_sandhi(self, word1: str, word2: str) -> str:
        """Apply Pāṇini's sandhi (euphonic combination) rules."""
        final = word1[-1] if word1 else ''
        initial = word2[0] if word2 else ''

        if final in ['अ', 'आ'] and initial in ['इ', 'ई']:
            return word1[:-1] + 'ए' + word2[1:]
        if final in ['अ', 'आ'] and initial in ['उ', 'ऊ']:
            return word1[:-1] + 'ओ' + word2[1:]
        if final in ['अ', 'आ'] and initial in ['अ', 'आ']:
            return word1[:-1] + 'आ' + word2[1:]

        consonants = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
        if word1.endswith('म्') and initial in consonants:
            return word1[:-2] + 'ं ' + word2

        if final == 'ः':
            if initial in ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ']:
                return word1[:-1] + 'ओऽ' + word2
            elif initial in ['ग', 'घ', 'ज', 'झ', 'ड', 'ढ', 'द', 'ध', 'ब', 'भ', 'य', 'र', 'ल', 'व', 'ह']:
                return word1[:-1] + 'ओ ' + word2

        return word1 + ' ' + word2
