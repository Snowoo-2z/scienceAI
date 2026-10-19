import tempfile
import unittest
import numpy as np
from data import dataset
from experiment import features,loss_grad,calibrate,predict,build_library,programs
from typed import arguments,masked,challenge

class RoutingTests(unittest.TestCase):
    def test_splits_disjoint(self):
        d=dataset();sets=[{r['text'] for r in d[s]} for s in ('train','validation','test')]
        self.assertEqual([len(d[s]) for s in ('train','validation','test')],[120,68,136])
        for i in range(3):
            for j in range(i):self.assertFalse(sets[i]&sets[j])
        self.assertFalse({r['text'] for r in challenge()}&set.union(*sets))

    def test_features_normalized_deterministic_unicode(self):
        a=features(['ÉCHANGE','e\u0301change',''])
        np.testing.assert_array_equal(a[0],a[1])
        np.testing.assert_array_equal(a,features(['ÉCHANGE','e\u0301change','']))
        np.testing.assert_allclose(np.linalg.norm(a[:2],axis=1),1,atol=1e-6)
        self.assertTrue(np.isfinite(a).all())

    def test_gradient(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(5,7));y=np.array([0,1,2,1,0])
        w=rng.normal(size=(7,3));b=rng.normal(size=3)
        _,dw,db=loss_grad(x,y,w,b)
        for arr,g in ((w,dw),(b,db)):
            for index in np.ndindex(arr.shape):
                value=arr[index];arr[index]=value+1e-6;plus=loss_grad(x,y,w,b)[0]
                arr[index]=value-1e-6;minus=loss_grad(x,y,w,b)[0];arr[index]=value
                self.assertAlmostEqual(g[index],(plus-minus)/2e-6,places=7)

    def test_calibration_and_reject(self):
        labels=np.zeros(20,dtype=int);truth=labels.copy();truth[0]=-1
        c=calibrate(labels,np.ones(20)*.8,truth)
        self.assertEqual(c['accepted'],20);self.assertEqual(c['wrong'],1)
        truth[1]=-1
        self.assertGreater(calibrate(labels,np.ones(20)*.8,truth)['threshold'],1)
        self.assertEqual(calibrate(np.full(20,-1),np.ones(20),np.full(20,-1))['accepted'],0)
        w=np.zeros((512,9),np.float32);b=np.zeros(9,np.float32);b[8]=10
        self.assertEqual(predict((w,b),features(['bonjour']))[0][0],-1)

    def test_supplied_binding_not_general_parser(self):
        self.assertEqual(arguments('fais un échange entre quatre et cinq'),3)
        self.assertEqual(arguments('inverse 9 et 8'),7)
        self.assertIsNone(arguments('échange 1 et 16'))
        self.assertEqual(masked('deux et 3'),' #  et  # ')
        # Explicit counterexample: last-two binding cannot parse two operations.
        self.assertEqual(arguments('échange 1 et 2 puis 3 et 4'),2)

    def test_no_oracle_fallback_and_budget(self):
        model=(np.zeros((512,8),np.float32),np.zeros(8,np.float32))
        rows=[dict(text='a deliberately opaque instruction',label=0)]
        seen=[]
        def wrong_route(texts):
            seen.extend(texts);return np.array([1]),np.array([1.])
        with tempfile.TemporaryDirectory() as root:
            build_library(root,71)
            r=programs(model,rows,0,root,65536,71,'top1',program_count=2,route=wrong_route)
            self.assertEqual(r[0]['correct'],0);self.assertTrue(seen)
            self.assertTrue(all(t==rows[0]['text'] for t in seen))
            self.assertTrue(all(a['peak_router_plus_operator_payload']<=65536 for a in r))
            r=programs(model,rows,1.1,root,65536,71,'selective',program_count=2,route=wrong_route)
            self.assertTrue(all(a['applied_steps']==0 for a in r))
            r=programs(model,rows,0,root,65536,71,'oracle',program_count=2)
            self.assertTrue(all(a['correct']==2 for a in r))
            with self.assertRaises(ValueError):programs(model,rows,0,root,16,71,'top1',program_count=2)

if __name__=='__main__':unittest.main()
