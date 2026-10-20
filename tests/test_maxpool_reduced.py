"""Compare reduced formulations with full MILP and native PyTorch."""
import cvxpy as cp
import numpy as np
import pytest
import torch
from torch import nn
from ncet import Bounds, form_milp


def fixture(kind):
    shape = (1, 2, 4)
    lower = np.zeros(shape)
    upper = np.ones(shape)
    pool = nn.Sequential(nn.MaxPool2d(2)).eval()
    if kind in {'fixed', 'mixed'}:
        lower[0,0,0] = 2; upper[0,0,0] = 3
    if kind == 'fixed':
        lower[0,0,2] = 2; upper[0,0,2] = 3
    if kind == 'mixed':
        lower[0,0,2] = 2; upper[0,0,2] = 3
        lower[0,1,2] = 1; upper[0,1,2] = 4
    if kind == 'ties':
        lower.fill(-1); upper.fill(-1)
    if kind == 'padding':
        pool = nn.Sequential(nn.MaxPool2d(2,stride=1,padding=1)).eval()
        lower -= 3; upper -= 3
    return pool, lower, upper


@pytest.mark.parametrize('kind', ['fixed','mixed','uncertain','ties','padding'])
@pytest.mark.parametrize('fraction', [0., .37, 1.])
def test_fixed_input_matches_pytorch(kind, fraction):
    model, lower, upper = fixture(kind)
    sample = lower + fraction*(upper-lower)
    expected = model(torch.tensor(sample,dtype=torch.float64).unsqueeze(0)).squeeze(0).numpy()
    enc = form_milp(model,Bounds(lower,upper),maxpool_binary_mode='reduced')
    problem = cp.Problem(cp.Minimize(0),enc.constraints+[next(iter(enc.inputs.values()))==sample])
    problem.solve(solver='HIGHS')
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(enc.outputs[0].value,expected,atol=1e-7)


@pytest.mark.parametrize('kind,expected', [('fixed',0),('mixed',2),('uncertain',8),('ties',8)])
def test_expected_binary_reductions(kind,expected):
    model,lower,upper=fixture(kind)
    full=form_milp(model,Bounds(lower,upper))
    reduced=form_milp(model,Bounds(lower,upper),maxpool_binary_mode='reduced')
    assert full.stats.binary_variables==8
    assert reduced.stats.binary_variables==expected


@pytest.mark.parametrize('kind', ['fixed','mixed','uncertain','ties','padding'])
@pytest.mark.parametrize('sign', [-1,1])
def test_global_extremum_matches_full_integer_formulation(kind,sign):
    model,lower,upper=fixture(kind)
    objectives=[]
    for mode in ('full','reduced'):
        enc=form_milp(model,Bounds(lower,upper),maxpool_binary_mode=mode)
        # Mixed signs expose both minimization and maximization of different outputs.
        weights=np.linspace(-.8,1.2,enc.outputs[0].size).reshape(enc.outputs[0].shape)
        p=cp.Problem(cp.Minimize(sign*cp.sum(cp.multiply(weights,enc.outputs[0]))),enc.constraints)
        p.solve(solver='HIGHS')
        assert p.status==cp.OPTIMAL
        objectives.append(p.value)
    assert objectives[0]==pytest.approx(objectives[1],abs=1e-7)


def test_invalid_maxpool_mode_rejected():
    model,lower,upper=fixture('uncertain')
    with pytest.raises(ValueError,match='maxpool_binary_mode'):
        form_milp(model,Bounds(lower,upper),maxpool_binary_mode='invalid')
