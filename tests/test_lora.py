import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lumina_lora import LoRALinear,adapter_state,apply_state

class LoRATests(unittest.TestCase):
    def test_zero_adapter_matches_base_and_base_stays_frozen(self):
        base=torch.nn.Linear(5,3);original=base.weight.detach().clone();layer=LoRALinear(base,rank=2,alpha=4)
        x=torch.randn(4,5)
        torch.testing.assert_close(layer(x),base(x))
        opt=torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.01)
        layer(x).square().mean().backward();opt.step()
        torch.testing.assert_close(base.weight,original)
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.lora_B.abs().sum().item(),0)
    def test_adapter_roundtrip_and_incompatible_rejected(self):
        layer=LoRALinear(torch.nn.Linear(4,3),rank=2);state=adapter_state(layer)
        apply_state(layer,state)
        with self.assertRaises(ValueError):apply_state(layer,{})

if __name__=='__main__':unittest.main()
