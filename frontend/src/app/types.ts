export type DataQuality = 'illustrative' | 'validated_holdout' | 'validated' | string;

export interface Forecast {
  id: string;
  as_of_date: string;
  target_date: string;
  horizon: 1 | 5;
  current_price_cents_per_lb: number;
  predicted_price_cents_per_lb: number;
  predicted_return_pct: number;
  direction: 'up' | 'down' | 'flat';
  actual_price_cents_per_lb: number | null;
  absolute_error: number | null;
  direction_correct: boolean | null;
  origin_type: 'live' | 'backtest';
  model_name: string;
  model_version: string;
  data_quality: DataQuality;
  generated_at: string;
}

export interface LatestForecast {
  forecasts: Forecast[];
  data_as_of: string;
  artifact_version: string;
  data_quality: DataQuality;
  stale: boolean;
}

export interface MarketPoint { date: string; series: string; close: number; }
export interface MarketHistory {
  points: MarketPoint[];
  data_as_of: string;
  source_note: string;
  data_quality: DataQuality;
}

export interface Contribution {
  feature: string;
  display_name: string;
  feature_value: number;
  contribution_pct: number;
}

export interface Explanation {
  forecast_id: string;
  horizon: number;
  model_name: string;
  explainer: string;
  base_value_pct: number;
  predicted_return_pct: number;
  contributions: Contribution[];
  approximation_error_pct?: number;
  equation: string;
}

export interface ModelMetric {
  model: string;
  horizon: number;
  mae: number;
  rmse: number;
  mape: number;
  directional_accuracy: number;
  selected: boolean;
  data_quality: DataQuality;
  walkforward?: {
    mae: number; rmse: number; mape: number; directional_accuracy: number;
    balanced_accuracy: number; majority_direction_accuracy: number;
    sample_count: number; mae_ci_95: number[];
  } | null;
  parameters?: Record<string, string | number> | null;
  training_history?: { epoch: number; loss: number; val_loss: number }[] | null;
}

export interface ModelEvaluation {
  artifact_version: string | null;
  note: string;
  selection_audit: Record<string, { locked_candidate?: string; selected?: string; fold_wins_vs_naive?: Record<string, number> }> | null;
  walkforward_report: {
    folds: { fold: number; test_start: string; test_end: string; sample_count: number; metrics: Record<string, { mae: number; directional_accuracy: number }> }[];
    feature_ablation: { fold: number; horizon: number; cotton_mae: number; cotton_macro_mae: number; full_mae: number }[];
    cftc_candidate: string;
  } | null;
}

export interface SimulationAdjustments {
  dxy_pct_change: number;
  wti_pct_change: number;
  cftc_net_delta_contracts: number;
  volatility_multiplier: number;
}

export interface SimulationResult {
  horizon: number;
  baseline_price_cents_per_lb: number;
  scenario_price_cents_per_lb: number;
  delta_cents_per_lb: number;
  delta_pct: number;
  model_name: string;
  baseline_kind?: 'production' | 'experimental';
}

export interface SimulationResponse {
  id: string;
  as_of_date: string;
  results: SimulationResult[];
  disclaimer: string;
  model_version: string;
}

export interface ReplayResponse { as_of_date: string; forecasts: Forecast[]; }
