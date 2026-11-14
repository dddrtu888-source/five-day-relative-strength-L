"""Synthetic checks for the public five-day research module."""
import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
SPEC=importlib.util.find_spec('relative_strength_l.core') if importlib.util.find_spec('relative_strength_l') else None
core=None
if SPEC is not None:
 from relative_strength_l import core

class ResearchContracts(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(core,'Public five-day core has not been extracted yet')
 def test_leave_one_out_ties_and_sample_sd(self):
  r=pd.DataFrame([[0.,.01,.01,.03]])
  p,z,l=core.distribution_scores(r,r.notna(),min_peers=2)
  np.testing.assert_allclose(p.iloc[0],[0.,.5,.5,1.])
  self.assertAlmostEqual(l.iloc[0,3],.03-(0+.01+.01)/3)
  self.assertAlmostEqual(z.iloc[0,3],l.iloc[0,3]/np.std([0,.01,.01],ddof=1))
 def test_weights_match_frozen_paper(self):
  weights=core.candidate_weights(1000,20261112)
  self.assertEqual(weights.shape,(1002,5));self.assertTrue((weights>=.05).all())
  np.testing.assert_allclose(weights.sum(axis=1),1)
  np.testing.assert_allclose(weights[227],[.05000495819755214,.05019117310110686,.050254482568096295,.050230980173204334,.7993184059600403])
 def test_score_window_order_and_missing_no_fill(self):
  s=pd.DataFrame(np.arange(7.)[:,None]);s.iloc[1]=np.nan
  x=core.score_windows(s)
  self.assertTrue(np.isnan(x[:4]).all());self.assertTrue(np.isnan(x[4,0,1]))
  np.testing.assert_array_equal(x[6,0],[2,3,4,5,6])
 def test_selection_ties_and_missing(self):
  x=np.array([[2.,2.,2.,1.],[np.nan,3.,np.nan,1.]])
  np.testing.assert_array_equal(core.top_indices(x,2),[[0,1],[1,3]])
 def test_future_peak_includes_entry_day_requires_full_window(self):
  o=pd.DataFrame([100.]*6);h=pd.DataFrame([999.,110.,120.,105.,101.,104.])
  p=core.peak_returns(h,o,3)
  self.assertAlmostEqual(p.iloc[0,0],.2);self.assertTrue(np.isnan(p.iloc[-2,0]))
  h.iloc[2]=np.nan;self.assertTrue(np.isnan(core.peak_returns(h,o,3).iloc[0,0]))
 def test_training_label_isolation_under_future_poison(self):
  dates=pd.bdate_range('2021-08-01','2026-11-11');sp=core.temporal_split(dates)
  rng=np.random.default_rng(123);o=pd.DataFrame(rng.uniform(90,110,(len(dates),4)),index=dates);h=o*1.1
  windows=np.ones((len(dates),4,5));a=core.training_opportunities(windows,o,h,sp)
  h.loc[sp['test']]=1e12;o.loc[sp['test']]=.00001
  b=core.training_opportunities(windows,o,h,sp)
  for k in a:np.testing.assert_allclose(a[k],b[k],equal_nan=True)
 def test_shortlist_rank_does_not_use_test_metrics(self):
  trials=pd.DataFrame({'编号':[0,1], '训练目标':[.02,.01], '测试收益':[-100,100]})
  self.assertEqual(core.rank_trials(trials)['编号'].tolist(),[0,1])

if __name__=='__main__':unittest.main()
