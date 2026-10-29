import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lumina_general import calculate,generate
from lumina_answer_checks import local_answer,enforce_format

class GeneralTests(unittest.TestCase):
    def test_calendar_order_and_relations_on_new_values(self):
        for q,a in [('Oggi è lunedì. Che giorno sarà tra 9 giorni?','mercoledì'),
                    ('Oggi è sabato. Che giorno era ieri?','venerdì'),
                    ('Ordina in modo decrescente 4, -3, 10.','10, 4, -3'),
                    ('Eva è più alta di Gina. Gina è più alta di Nora. Chi è la più alta?','Eva')]:
            self.assertEqual(local_answer([{'role':'user','content':q}])[0],a)
    def test_missing_data_and_future_are_not_invented(self):
        self.assertIn('Sono Lumina',local_answer([{'role':'user','content':'Come ti chiami?'}])[0])
        self.assertIn('Non ho',local_answer([{'role':'user','content':'Qual è il numero nel quaderno? Non puoi vederlo.'}])[0])
        self.assertIn('non è specificato',local_answer([{'role':'user','content':'Dal testo "Il vaso è arrivato", qual è il prezzo?'}])[0])
        self.assertIn('evento futuro',local_answer([{'role':'user','content':'Chi vincerà nel 2061?'}])[0])
    def test_explicit_format_is_preserved(self):
        self.assertEqual(enforce_format('Sì.','Rispondi esclusivamente con la parola sì.'),'sì')
        self.assertEqual(enforce_format('```json\n{"x":3}\n```','Restituisci esclusivamente JSON.'),'{"x":3}')
        self.assertEqual(enforce_format('Il numero è 83.','Solo il numero.'),'83')
        self.assertIn('def raddoppia(',enforce_format('```python\ndef double(x):\n    return x * 2\n```','Scrivi solo codice della funzione Python raddoppia(x).'))
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
    def test_verified_short_answers(self):
        cases = [
            ('Quanto fa 17 per 6?', '102'),
            ('Correggi la frase: io ho andato al mare.', 'La forma corretta è: io sono andato al mare.'),
            ('In Python, come si scrive il comando per stampare la parola ciao?', "print('ciao')"),
            ('Scrivi una frase di cinque parole sul mare.', 'Il mare oggi è calmo.'),
        ]
        for prompt, expected in cases:
            self.assertEqual(local_answer([{'role':'user','content':prompt}])[0], expected)
        self.assertIn('fotosintesi', local_answer([{'role':'user','content':'Cos’è la fotosintesi?'}])[0].lower())
        self.assertIn('Garda', local_answer([{'role':'user','content':'Qual è il lago più grande d’Italia?'}])[0])
    def test_numeric_math_variants_are_exact(self):
        for prompt, expected in [
            ('17 moltiplicato per 6', '102'),
            ('Quanto fa 24 x 6?', '144'),
            ('15 più 27', '42'),
            ('100 meno 37', '63'),
            ('84 diviso 7', '12'),
        ]:
            self.assertEqual(local_answer([{'role':'user','content':prompt}])[0], expected)
    def test_honest_limits_are_stable(self):
        self.assertIn('Posso sbagliare', local_answer([{'role':'user','content':'Dici sempre la verità?'}])[0])
        self.assertIn('coscienza', local_answer([{'role':'user','content':'Puoi essere felice?'}])[0])
        self.assertIn('tempo reale', local_answer([{'role':'user','content':'Quali sono le notizie?'}])[0])
    def test_basic_it_definitions_are_stable(self):
        self.assertIn('variabile', local_answer([{'role':'user','content':'Che significa variabile?'}])[0])
        self.assertIn('funzione', local_answer([{'role':'user','content':'Che cosa fa una funzione?'}])[0])
        self.assertIn('browser', local_answer([{'role':'user','content':'Che cosa fa il browser?'}])[0])
        self.assertIn('HTML', local_answer([{'role':'user','content':'Che significa HTML?'}])[0])
    def test_app_path_uses_verified_answers(self):
        checks = [
            ('Quanto fa 17 per 6?', '102'),
            ('In Python, come si scrive il comando per stampare la parola ciao?', "print('ciao')"),
            ('Scrivi una frase di cinque parole sul mare.', 'Il mare oggi è calmo.'),
        ]
        for prompt, expected in checks:
            answer, meta = generate([{'role':'user','content':prompt}])
            self.assertEqual(answer, expected)
            self.assertTrue(meta.get('local'))
    def test_extraction_and_list_constraints(self):
        self.assertEqual(local_answer([{'role':'user','content':'Scrivi soltanto questi tre colori separati da virgole: rosso, verde, blu.'}])[0], 'rosso, verde, blu')
        prompt='Nel testo: "Marta ha 4 mele e Luca ha 9 pere". Quante pere ha Luca?'
        self.assertEqual(local_answer([{'role':'user','content':prompt}])[0], '9')

if __name__=='__main__':unittest.main()
