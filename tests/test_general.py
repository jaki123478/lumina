import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_general import calculate

class GeneralTests(unittest.TestCase):
    def test_calculator_new_values_and_precedence(self):
        for q,a in [('Calcola 36 + 19. Solo numero.','55'),('Quanto fa 23 per 7?','161'),
                    ('Calcola (5 + 3) * 7','56'),('Quanto fa 2 + 2?','4'),
                    ('Calcola 1.5 + 2.25','3.75'),('Quanto fa -9 + 4?','-5'),
                    ('Calcola 84 / 7.','12')]:
            self.assertEqual(calculate(q),a)
    def test_does_not_evaluate_python_or_partial_expressions(self):
        for q in ['Qual è la capitale della Francia?','Calcola __import__("os")',
                  'Calcola 2 + 2e3','Calcola 1 / 0','Calcola 9 ** 999999','Calcola (2+3).real']:
            self.assertIsNone(calculate(q),q)

if __name__=='__main__':unittest.main()
