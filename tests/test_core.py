import unittest
import numpy as np
import torch
from reid.clustering import refine
from reid.evaluate import retrieval
from reid.losses import global_embedding,cluster_triplet

class CoreTests(unittest.TestCase):
    def test_market_exclusions(self):
        q=np.array([[1.,0.]])
        g=np.array([[1.,0.],[.9,.1],[.8,.2],[.7,.3]])
        qm={'person_ids':[1],'camera_ids':[1]}
        gm={'person_ids':[1,-1,2,1],'camera_ids':[1,2,2,2]}
        metrics,_=retrieval(q,g,qm,gm)
        self.assertEqual(metrics['Rank-1'],0)
        self.assertEqual(metrics['Rank-5'],1)
        self.assertEqual(metrics['mAP'],.5)
    def test_outlier_inherits_without_changing_clustered(self):
        d=np.array([[0,.1,.9],[.1,0,.8],[.9,.8,0]])
        labels,stats=refine(d,np.array([-1,0,1]),s=1)
        np.testing.assert_array_equal(labels,[0,0,1])
        self.assertEqual(stats['recovered'],1)
    def test_high_confidence_stops_lower_tiers_at_half_budget(self):
        d=np.array([[0,.5,.1],[.5,0,.4],[.1,.4,0]])
        labels,stats=refine(d,np.array([-1,-1,0]),s=2,budget=2)
        np.testing.assert_array_equal(labels,[0,-1,0])
        self.assertEqual(stats['recovered'],1)
    def test_low_confidence_requires_consensus(self):
        d=np.array([[0,.6,.61],[.6,0,.1],[.61,.1,0]])
        labels,_=refine(d,np.array([-1,0,1]),s=2)
        self.assertEqual(labels[0],-1)
    def test_gel_aggregates_and_ignores_outlier(self):
        x=torch.tensor([[1.,0.]],requires_grad=True)
        mem=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[100.,100.]])
        loss=global_embedding(x,mem,torch.tensor([0,0,1,-1]),torch.tensor([0]),temperature=1)
        expected=-np.log(2*np.e/(2*np.e+1))
        self.assertAlmostEqual(float(loss.detach()),expected,places=6)
        loss.backward(); self.assertTrue(torch.isfinite(x.grad).all())
    def test_triplet_finite_with_valid_and_invalid_batches(self):
        x=torch.nn.functional.normalize(torch.randn(4,8),dim=1).requires_grad_()
        loss=cluster_triplet(x,torch.tensor([0,0,1,1]),x.detach()[[0,2]])
        loss.backward(); self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(cluster_triplet(x,torch.tensor([0,0,0,0]),x[:1])),0)

if __name__=='__main__': unittest.main()
