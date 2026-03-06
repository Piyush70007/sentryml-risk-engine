from monitoring.drift_daily import rel_change

def test_rel_change_20_percent():
    assert rel_change(100, 120) == 0.2

def test_rel_change_5_percent():
    assert rel_change(200, 210) == 0.05

def test_rel_change_zero_zero():
    assert rel_change(0, 0) == 0.0