import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_public_web import Extractor

class ResearchTests(unittest.TestCase):
    def test_article_paragraphs_preserve_inline_words(self):
        p=Extractor();p.feed('<nav>menu</nav><article><p>Due <b>satelliti</b> nuovi.</p><script>esegui</script></article>')
        self.assertEqual(p.paragraphs,['Due satelliti nuovi.'])
        self.assertNotIn('esegui',p.parts)

if __name__=='__main__':unittest.main()
