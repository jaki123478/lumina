import sys,unittest,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_web_sources import parse_feed,latest,enabled

class WebSourcesTests(unittest.TestCase):
    def test_environment_offline_override(self):
        with patch.dict(os.environ,{'LUMINA_ALLOW_INTERNET':'0'}):self.assertFalse(enabled())
    def test_rdf_date_and_html_sanitization(self):
        xml=b'''<rdf:RDF xmlns:rdf="urn:rdf" xmlns:r="urn:rss" xmlns:dc="urn:dc"><r:item><r:title>A &amp; B</r:title><r:link>https://www.example.org/news</r:link><dc:date>2020-01-01T12:00:00Z</dc:date></r:item></rdf:RDF>'''
        rows=parse_feed(xml,'Prova','example.org');self.assertEqual(rows[0]['title'],'A & B')
    def test_rejects_external_links_and_missing_date(self):
        xml=b'<rss><channel><item><title>test</title><link>https://evil.example/x</link><pubDate>Wed, 01 Jan 2020 12:00:00 GMT</pubDate></item><item><title>test</title><link>https://example.org/x</link></item></channel></rss>'
        self.assertEqual(parse_feed(xml,'Prova','example.org'),[])
    def test_revoked_consent_prevents_network(self):
        with patch('lumina_web_sources.enabled',return_value=False),patch('requests.Session') as session:
            with self.assertRaises(PermissionError):latest()
            session.assert_not_called()

if __name__=='__main__':unittest.main()
