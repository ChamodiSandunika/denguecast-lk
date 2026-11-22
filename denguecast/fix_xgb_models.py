"""Re-save Colab-trained pickles that contain XGBoost boosters so they load locally.
The saved booster buffer has a Config block this xgboost build rejects; the Model block loads fine."""
import joblib, xgboost as xgb

def _setstate(self, state):
    buf = bytes(state["handle"])
    j = buf.find(b"Model")
    self.handle = None
    b = xgb.Booster()
    b.load_model(bytearray(buf[j + 5:-1]))
    self.handle = b.handle
    b.handle = None
    self.feature_names = state.get("feature_names")
    self.feature_types = state.get("feature_types")

orig = xgb.Booster.__setstate__
xgb.Booster.__setstate__ = _setstate
fixed = {}
for f in ["xgboost", "stacking"]:
    fixed[f] = joblib.load(f"models/{f}.pkl")
xgb.Booster.__setstate__ = orig
for f, m in fixed.items():
    joblib.dump(m, f"models/{f}.pkl", compress=3)
    print("re-saved", f)
