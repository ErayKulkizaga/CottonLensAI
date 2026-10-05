from dataclasses import dataclass
from pathlib import Path

# Ages include the mandatory one-Cotton-observation lag. This is a data-quality
# limit, not a predictive hyperparameter; three observations is the Day-1 default.
EXTERNAL_MAX_AGE_SESSIONS = 3
DATA_POLICY_VERSION = "cotton-observation-next-utc-day-v1"
HORIZON_SEMANTICS = "next_1_or_5_recorded_cotton_observations"

FEATURE_NAMES = [
    "cotton_ret_1",
    "cotton_ret_5",
    "cotton_ret_10",
    "cotton_ret_20",
    "cotton_momentum_5",
    "cotton_momentum_20",
    "cotton_sma_ratio_5_20",
    "cotton_sma_ratio_20_50",
    "cotton_volatility_5",
    "cotton_volatility_20",
    "cotton_range",
    "cotton_volume_change",
    "cotton_volume_z20",
    "cotton_volatility_regime_20_60",
    "dxy_ret_1",
    "dxy_ret_5",
    "dxy_ret_20",
    "wti_ret_1",
    "wti_ret_5",
    "wti_ret_20",
    "cotton_dxy_corr_60",
    "cotton_wti_corr_60",
    "month_sin",
    "month_cos",
]


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def mlruns(self) -> Path:
        return self.root / "mlruns"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def releases(self) -> Path:
        return self.root / "artifacts" / "releases"

    def create(self) -> None:
        for path in (self.raw, self.processed, self.mlruns, self.checkpoints, self.releases):
            path.mkdir(parents=True, exist_ok=True)
