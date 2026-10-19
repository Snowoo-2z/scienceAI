import tempfile
from pathlib import Path
import sqlite3
import unittest
import numpy as np
from library import Paged,build,pack,execute,error,truth


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.meta=build(cls.root,count=96,d=16,group_dictionary=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_codec(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(3,8))
        u=rng.normal(size=(8,2));v=rng.normal(size=(2,8))
        np.testing.assert_allclose(execute(pack(2,[u,v],2),x,8),x@(np.eye(8)+u@v).T)
        np.testing.assert_array_equal(execute(pack(0),x,8),x)
        t=rng.normal(size=(8,8))
        np.testing.assert_allclose(execute(pack(1,[t]),x,8),x@t.T)
        basis=rng.normal(size=(3,8,8));code=rng.normal(size=3)
        np.testing.assert_allclose(execute(pack(3,[code],3),x,8,basis),x@(np.eye(8)+np.einsum('r,rde->de',code,basis)).T)

    def test_numeric_quality_on_new_vectors(self):
        rng=np.random.default_rng(9000)
        for method in ('dense','adaptive','shared'):
            obj=Paged(self.root,method,65536)
            try:
                for op in range(self.meta['count']):
                    x=rng.normal(size=(5,16));expected=x@truth(op,16,40).T
                    relative=np.linalg.norm(obj.apply(x,op)-expected)/np.linalg.norm(expected)
                    self.assertLess(relative,1e-4)
            finally:obj.close()

    def test_cache_budget_and_hit(self):
        obj=Paged(self.root,'adaptive',4096)
        try:
            obj.get(1);obj.get(1)
            self.assertEqual(obj.hits,1)
            for i in range(self.meta['count']):
                obj.get(i)
                self.assertEqual(obj.used,sum(map(len,obj.cache.values())))
                self.assertLessEqual(obj.used+obj.fixed,4096)
            self.assertLessEqual(obj.peak,4096)
        finally:obj.close()

    def test_oversized_blob_not_cached(self):
        obj=Paged(self.root,'dense',64)
        try:
            obj.get(2);obj.get(2)
            self.assertEqual(obj.used,0)
            self.assertEqual(obj.misses,2)
            self.assertEqual(obj.hits,0)
        finally:obj.close()

    def test_shared_fixed_basis_counted(self):
        self.assertTrue(self.meta['shared_enabled'])
        fixed=self.meta['shared_basis_payload_bytes']
        with self.assertRaises(ValueError):Paged(self.root,'shared',fixed-1)
        obj=Paged(self.root,'shared',fixed+4096)
        try:
            for i in range(96):obj.get(i)
            self.assertEqual(obj.fixed,fixed)
            self.assertLessEqual(obj.peak,fixed+4096)
        finally:obj.close()

    def test_readonly_and_unknown(self):
        obj=Paged(self.root,'dense',4096)
        try:
            with self.assertRaises(KeyError):obj.get(999999)
            with self.assertRaises(sqlite3.OperationalError):obj.conn.execute('DELETE FROM operators')
        finally:obj.close()

    def test_determinism_and_disk_accounting(self):
        for method in ('dense','adaptive','shared'):
            a=Paged(self.root,method,65536);b=Paged(self.root,method,65536)
            try:
                for i in (1,2,3):
                    np.testing.assert_array_equal(a.apply(np.eye(16),i),b.apply(np.eye(16),i))
                self.assertEqual(a.logical_bytes,b.logical_bytes)
            finally:a.close();b.close()
        self.assertEqual((self.root/'dense.sqlite').stat().st_size,self.meta['database_file_bytes']['dense'])


if __name__=='__main__':unittest.main()
