import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cshap.cli import main
from cshap.config import load_config
from cshap.data import prepare_data


def test_prepare_breast_cancer_matches_notebook_split():
    d = prepare_data("breast_cancer", max_reference=None)
    assert d.X_train.shape == (455, 30) and d.X_test.shape == (114, 30)
    np.testing.assert_allclose(d.X_train.mean(axis=0), 0, atol=1e-10)
    assert d.X_ref.shape == d.X_train.shape


def test_config_merge(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("n_jobs: 3\nsweep: {seeds: 2}\n")
    cfg = load_config(p)
    assert cfg["n_jobs"] == 3 and cfg["sweep"]["seeds"] == 2 and cfg["sweep"]["sizes"]


def test_cli_explain_and_compare(tmp_path, toy, capsys):
    X, y = toy
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    joblib.dump(model, tmp_path / "m.joblib")
    cols = [f"f{i}" for i in range(X.shape[1])]
    pd.DataFrame(X[:200], columns=cols).to_csv(tmp_path / "bg.csv", index=False)
    pd.DataFrame(X[200:203], columns=cols).to_csv(tmp_path / "x.csv", index=False)
    out = tmp_path / "phi.csv"
    common = ["--model", str(tmp_path / "m.joblib"), "--background", str(tmp_path / "bg.csv"),
              "--data", str(tmp_path / "x.csv"), "--k", "5", "--nsamples", "64"]
    main(["explain", *common, "--output", str(out)])
    phi = pd.read_csv(out)
    assert list(phi.columns) == ["base_value", *cols] and len(phi) == 3
    recon = phi[cols].sum(axis=1) + phi["base_value"]
    np.testing.assert_allclose(recon, model.predict_proba(X[200:203])[:, 1], atol=1e-8)
    main(["compare", *common, "--reference-size", "50"])
    assert "cshap_k5" in capsys.readouterr().out
