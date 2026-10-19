from pathlib import Path
import tempfile
import unittest
from core import Space,execute,INVALID,key,bounded
from probes import generate
from tasks import TASKS,POOLS,TEST

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'space.sqlite';self.s=Space(self.path)
    def tearDown(self):self.s.close();self.temp.cleanup()

    def test_closed_bounded_interpreter(self):
        self.assertIs(execute(['__import__'],1),INVALID)
        self.assertIs(execute(['square'],10**9),INVALID)
        self.assertIs(execute(['lower'],True),INVALID)
        self.assertIs(execute(['sum'],['1','2']),INVALID)
        self.assertFalse(bounded([[1]]));self.assertFalse(bounded('x'*257))
        self.assertEqual(execute(['split_space','reverse','join_space'],'abc def'),'def abc')

    def test_ambiguity_not_collapsed(self):
        self.s.fit([dict(input=1,output=1)])
        ps=[p for _,p in self.s.programs()]
        self.assertIn([],ps);self.assertIn(['absolute'],ps)
        self.assertEqual(self.s.predict(-2)['status'],'ambiguous')
        self.assertEqual(self.s.shortest(-2)['output'],-2)

    def test_observation_exact_filter_and_persistence(self):
        self.s.fit([dict(input=1,output=3)])
        survivors={key(p) for _,p in self.s.programs() if key(execute(p,-4))==key(-7)}
        self.s.observe(-4,-7)
        self.assertEqual({key(p) for _,p in self.s.programs()},survivors)
        self.s.close();self.s=Space(self.path)
        self.assertEqual(self.s.predict(20)['output'],41)
        self.assertEqual(len(self.s.get('examples')),2)

    def test_contradiction_and_revise(self):
        self.s.fit([dict(input=' abc def ',output='abc def')],depth=1)
        self.s.observe('A B','a b');self.assertEqual(self.s.count(),0)
        revision=self.s.revise_if_empty(max_depth=2)
        self.assertEqual(revision['status'],'revised')
        self.assertEqual(self.s.predict('  MIXED WORDS  ')['output'],'mixed words')
        self.s.observe('A B','OTHER');self.assertEqual(self.s.count(),0)
        self.assertEqual(self.s.revise_if_empty(max_depth=3)['status'],'no_hypothesis')

    def test_incomplete_search_never_called_consensus(self):
        result=self.s.fit([dict(input=1,output=1)],max_nodes=1)
        self.assertFalse(result['complete'])
        self.assertEqual(self.s.predict(2)['status'],'incomplete_search')
        self.assertFalse(self.s.fit([dict(input=1,output=1)],max_matches=1)['complete'])

    def test_query_budget_and_no_repeat(self):
        self.s.fit([dict(input=1,output=1)])
        q=self.s.clarify([1,-2,3]);self.assertEqual(q['status'],'question');self.assertNotEqual(q['input'],1)
        self.assertEqual(self.s.clarify([-2],max_partition_bytes=1)['status'],'query_budget_exhausted')
        self.assertEqual(self.s.clarify([1])['status'],'no_disagreement_in_pool')

    def test_known_target_retained_and_consensus_sound_within_class(self):
        for name,kind,target,initial in TASKS[:-1]:
            self.s.fit([dict(input=x,output=execute(target,x)) for x in initial])
            self.assertTrue(self.s.get('search')['complete'])
            self.assertIn(target,[p for _,p in self.s.programs()])
            for x in POOLS[kind][:2]:self.s.observe(x,execute(target,x))
            self.assertIn(target,[p for _,p in self.s.programs()])
            for x in TEST[kind]:
                prediction=self.s.predict(x)
                if prediction['status']=='consensus':self.assertEqual(key(prediction['output']),key(execute(target,x)))

    def test_wrong_consensus_possible_outside_class(self):
        self.s.fit([dict(input=x,output=min(x,10)) for x in (-1,1,2,-9)])
        p=self.s.predict(20)
        self.assertEqual(p['status'],'consensus');self.assertEqual(p['output'],20)
        self.s.observe(20,10);self.assertEqual(self.s.count(),0)

    def test_copy_independent_and_probes_reproducible(self):
        self.s.fit([dict(input=1,output=3)])
        path=Path(self.temp.name)/'copy.sqlite';self.s.copy_to(path);other=Space(path)
        other.observe(1,4);self.assertEqual(other.count(),0);self.assertGreater(self.s.count(),0);other.close()
        self.assertEqual(generate('text',7),generate('text',7))
        self.assertEqual(len(generate('list',7)),64)

if __name__=='__main__':unittest.main()
