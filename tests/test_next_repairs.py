import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import ByteLevelBPETokenizer
from modeling_lumina_next import LuminaNextForCausalLM
from lumina_next_runtime import prompt_ids, stop_sequences, response_text, ROOT
from lumina_voice_neural import clean
from qwen_repair_trainer import curriculum, training_sequence, validate_teacher, RESERVED_QUESTIONS


class ScriptedModel(LuminaNextForCausalLM):
    def __init__(self,tokens):
        torch.nn.Module.__init__(self)
        self.tokens=tokens; self.step=0
    def forward(self,x,**kwargs):
        logits=torch.full((1,x.shape[1],32),-100.)
        logits[:,-1,self.tokens[self.step]]=100
        self.step+=1
        return logits, None


class NextRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=ByteLevelBPETokenizer(str(ROOT/'tokenizer_lumina_next/vocab.json'),str(ROOT/'tokenizer_lumina_next/merges.txt'))

    def test_stops_at_legacy_delimiter_without_inventing_next_turn(self):
        model=ScriptedModel([4,8,9,5])
        result=model.generate(torch.tensor([[3]]),temperature=0,max_new_tokens=4,stop_token_sequences=[[8,9]])
        self.assertEqual(result.tolist(),[[3,4,8,9]])
        self.assertEqual(model.step,3)

    def test_eos_one_is_not_padding(self):
        model=ScriptedModel([2,1,5])
        self.assertEqual(model.generate(torch.tensor([[3]]),temperature=0,max_new_tokens=3).tolist(),[[3,2,1]])

    def test_history_reaches_prompt(self):
        messages=[{'role':'user','content':'Il mio gatto è Zefiro.'},{'role':'assistant','content':'Va bene.'},{'role':'user','content':'Come si chiama?'}]
        text=self.tok.decode(prompt_ids(self.tok,messages))
        self.assertIn('Zefiro',text)
        self.assertIn('Come si chiama?',text)

    def test_long_last_message_is_not_silently_truncated(self):
        with self.assertRaises(ValueError): prompt_ids(self.tok,[{'role':'user','content':'a b '*500}],budget=50)

    def test_voice_drops_future_dialog_and_thoughts(self):
        self.assertEqual(clean('<think>segreto</think>Ciao.<|im_end|><|im_start|>user\naltro'),'Ciao.')
        self.assertNotIn('return',clean('Ecco:\n```python\ndef f(): return 1\n```'))
        self.assertNotIn('segreto',clean('Ciao.<think>segreto'))

    def test_teacher_rejects_incomplete_or_invalid_code(self):
        self.assertFalse(validate_teacher('```python\ndef f():'))
        self.assertFalse(validate_teacher('```python\ndef f():\n```'))
        self.assertFalse(validate_teacher('testo',finished=False))
        self.assertTrue(validate_teacher('```python\ndef f():\n    return 1\n```'))

    def test_no_training_validation_overlap_and_no_prompt_loss(self):
        train,valid=curriculum(73)
        import json
        key=lambda r: json.dumps(r['messages'],sort_keys=True)
        self.assertFalse({key(r) for r in train}&{key(r) for r in valid})
        self.assertFalse({r['messages'][-1]['content'] for r in train}&RESERVED_QUESTIONS)
        x,y=training_sequence(self.tok,train[0])
        self.assertEqual(len(x),len(y))
        prefix=prompt_ids(self.tok,train[0]['messages'])
        self.assertTrue(all(v==-100 for v in y[:len(prefix)-1]))
        self.assertEqual(y[-1],1)

    def test_missing_model_is_not_random_weights(self):
        import tempfile,json
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as tmp:
            cfg={'vocab_size':32,'d_model':16,'n_layers':1,'n_heads':2,'n_kv_heads':1,'head_dim':8,'d_ff':32,'max_position_embeddings':64}
            Path(tmp,'config.json').write_text(json.dumps(cfg))
            with self.assertRaises(FileNotFoundError): LuminaNextForCausalLM.from_pretrained(tmp)

    def test_next_cached_decoding_matches_full_context(self):
        cfg={'vocab_size':32,'d_model':16,'n_layers':2,'n_heads':2,'n_kv_heads':1,'head_dim':8,'d_ff':32,'max_position_embeddings':64}
        torch.manual_seed(9)
        m=LuminaNextForCausalLM(cfg).eval()
        ids=torch.tensor([[3,4,5,6]])
        with torch.inference_mode():
            _,cache=m(ids[:,:-1],use_cache=True)
            cached,_=m(ids[:,-1:],past_key_values=cache,use_cache=True)
            full=m(ids)
        torch.testing.assert_close(cached[:,-1],full[:,-1],atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
