import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_ricerca_vocale import needs_confirmation
from lumina_voice_neural import voice_settings
from lumina_voice_neural import synthesize_bytes

class VoiceQueryTests(unittest.TestCase):
    def test_clear_search_continues(self):self.assertFalse(needs_confirmation('cerca notizie sui satelliti galileo',.94))
    def test_unclear_audio_is_not_searched_blindly(self):self.assertTrue(needs_confirmation('cerca notizie',.43))
    def test_long_unstructured_text_requires_confirmation(self):
        self.assertTrue(needs_confirmation('vento barca finestra tavolo lampada corridoio matita ponte strada rumore fiore cielo',.95))

    def test_voice_profile_is_fast_and_bounded(self):
        settings = voice_settings()
        self.assertEqual(settings['length_scale'], 1.0)
        self.assertGreaterEqual(settings['volume'], .1)
        self.assertLessEqual(settings['volume'], .95)

    def test_neural_voice_returns_valid_wav(self):
        audio = synthesize_bytes('Ciao, sono Lumina.')
        self.assertGreater(len(audio), 1000)
        self.assertEqual(audio[:4], b'RIFF')
        self.assertEqual(audio[8:12], b'WAVE')

if __name__=='__main__':unittest.main()
