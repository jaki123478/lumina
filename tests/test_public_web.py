import unittest,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_public_web import validate_url,fetch,Extractor

class PublicWebTests(unittest.TestCase):
    def test_private_and_credentials_rejected(self):
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('127.0.0.1',443))]):
            with self.assertRaises(ValueError):validate_url('https://localhost/')
        for url in ['http://example.com','file:///tmp/x','https://user:pass@example.com','https://example.com:99/']:
            with self.assertRaises(ValueError):validate_url(url)
    def test_revoked_access_does_not_request(self):
        with patch('lumina_public_web.get_network_policy') as policy,patch('requests.Session') as session:
            policy.return_value.is_internet_allowed.return_value=False
            with self.assertRaises(PermissionError):fetch('https://example.com')
            session.assert_not_called()
    def test_scripts_are_not_read_as_text(self):
        p=Extractor();p.feed('<title>Titolo</title><script>password</script><p>Testo pubblico</p>')
        self.assertNotIn('password',p.parts);self.assertIn('Testo pubblico',p.parts)

if __name__=='__main__':unittest.main()
