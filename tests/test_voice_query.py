import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_ricerca_vocale import needs_confirmation

class VoiceQueryTests(unittest.TestCase):
    def test_clear_search_continues(self):self.assertFalse(needs_confirmation('cerca notizie sui satelliti galileo',.94))
    def test_unclear_audio_is_not_searched_blindly(self):self.assertTrue(needs_confirmation('cerca notizie',.43))
    def test_long_unstructured_text_requires_confirmation(self):
        self.assertTrue(needs_confirmation('vento barca finestra tavolo lampada corridoio matita ponte strada rumore fiore cielo',.95))

if __name__=='__main__':unittest.main()
