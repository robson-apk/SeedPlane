import unittest
from routing import *
class RoutingTests(unittest.TestCase):
 def test_modulo_equivalence(self):
  for a in range(96):
   for b in range(96):self.assertEqual(complement(a,b)>.5,a%32==b%32)
 def test_version_and_duplicate(self):
  e=(1,2,3,4,5,6)
  for mode in ['exact_envelope','seedplane_v5']:
   seen=set();self.assertTrue(accept(mode,e,e,seen));self.assertFalse(accept(mode,e,e,seen))
   for j in range(6):
    bad=list(e);bad[j]+=1;self.assertFalse(accept(mode,tuple(bad),e,set()))
 def test_seed_matches_exact(self):
  import random
  rng=random.Random(71)
  for _ in range(1000):
   e=tuple(rng.randrange(200) for _ in range(6));a=list(e)
   if rng.random()<.7:a[rng.randrange(6)]+=1
   self.assertEqual(accept('exact_envelope',tuple(a),e,set()),accept('seedplane_v5',tuple(a),e,set()))
if __name__=='__main__':unittest.main()
