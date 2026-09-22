import { Component, computed, inject, signal } from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';
import { forkJoin } from 'rxjs';

import { ApiService } from '../api.service';
import { ChartComponent } from '../chart.component';
import { ModelEvaluation, ModelMetric } from '../types';

@Component({
  standalone: true,
  imports: [ChartComponent],
  template: `
    <section class="page">
      <header class="page-head"><div><p class="eyebrow">Rolling-origin evidence</p><h1>Model lab</h1></div><div class="policy-note">Persistence (Naive) is the MAE baseline. A learned model needs ≥5% lower MAE, ≥3/4 fold wins, and direction ≥53% for T+1 or ≥55% for T+5.</div></header>
      @if (loading()) { <div class="state-panel"><span class="loader"></span><p>Loading experiment evidence…</p></div> }
      @else if (error()) { <div class="state-panel error"><strong>Metrics unavailable</strong><p>{{ error() }}</p></div> }
      @else {
        @if (isFixture()) { <div class="fixture-banner"><strong>Development fixture</strong><span>Metric values are illustrative placeholders, never production claims.</span></div> }
        @if (!evaluation()?.walkforward_report && !isFixture()) { <div class="backtest-banner"><strong>Legacy evaluation</strong><span>This artifact has no walk-forward evidence. Its historical numbers must not be presented as independent validation.</span></div> }
        @if (evaluation()?.walkforward_report) { <div class="backtest-banner"><strong>Historical audit</strong><span>Selection used four earlier rolling-origin periods. The 2024–2026 audit was previously observed and can only reject a locked candidate.</span></div> }
        <div class="model-summary">
          @for (horizon of [1, 5]; track horizon) {
            <article class="winner-card">
              <span>T+{{ horizon }} {{ isFixture() ? 'fixture selection' : 'selected model' }}</span>
              <strong>{{ selected(horizon)?.model }}</strong>
              <p>{{ selected(horizon)?.walkforward ? 'Walk-forward MAE' : 'Historical MAE' }} {{ (selected(horizon)?.walkforward?.mae ?? selected(horizon)?.mae)?.toFixed(2) }} ¢/lb</p>
              @if (selected(horizon)?.walkforward; as evidence) {
                <div class="gate-line"><span>MAE {{ improvement(horizon).toFixed(1) }}% vs Naive</span><span>Direction {{ evidence.directional_accuracy.toFixed(1) }}%</span><span>{{ decision(horizon)?.fold_wins_vs_naive?.[decision(horizon)?.locked_candidate ?? ''] ?? 0 }}/4 folds</span></div>
              } @else { <p>Walk-forward evidence unavailable for this artifact.</p> }
              @if (decision(horizon); as audit) { <p>{{ audit.locked_candidate === audit.selected ? 'Walk-forward choice retained' : 'Historical audit fallback to Naive' }}</p> }
            </article>
          }
          @if (!isFixture()) { <article class="policy-card"><span>Selection policy · not a measured result</span><strong>Weak candidates fall back to the Naive baseline.</strong></article> }
        </div>
        <article class="panel metric-panel">
          <div class="panel-head"><div><p class="card-label">{{ evaluation()?.walkforward_report ? 'Four equal-date folds' : 'Artifact-reported historical period' }}</p><h2>Performance matrix</h2></div><span class="origin-chip">Lower error is better</span></div>
          <div class="metric-table" role="table" aria-label="Model metrics">
            <div class="metric-row metric-header" role="row"><span>Model</span><span>Horizon</span><span>MAE</span><span>RMSE</span><span>MAPE</span><span>Direction</span></div>
            @for (metric of metrics(); track metric.model + metric.horizon) {
              <div class="metric-row" [class.selected]="metric.selected" role="row">
                <span><strong>{{ metric.model }}</strong>@if (metric.selected) { <small>Selected</small> }</span>
                <span>T+{{ metric.horizon }}</span>
                <span>{{ (metric.walkforward?.mae ?? metric.mae).toFixed(2) }}</span>
                <span>{{ (metric.walkforward?.rmse ?? metric.rmse).toFixed(2) }}</span>
                <span>{{ (metric.walkforward?.mape ?? metric.mape).toFixed(2) }}%</span>
                <span>{{ (metric.walkforward?.directional_accuracy ?? metric.directional_accuracy).toFixed(1) }}%</span>
              </div>
            }
          </div>
        </article>
        @if (evaluation()?.walkforward_report; as report) {
          <div class="evidence-grid">
            <article class="panel evidence-panel">
              <div class="panel-head"><div><p class="card-label">Pre-audit · 126 sessions each</p><h2>Fold ledger</h2></div></div>
              @for (fold of report.folds; track fold.fold) {
                <div class="evidence-row"><strong>Fold {{ fold.fold }}</strong><span>{{ fold.test_start }} → {{ fold.test_end }}</span><span>n={{ fold.sample_count }}</span></div>
              }
              <p class="evidence-footnote">Targets crossing a fold boundary are purged. Transformations are fitted on each fold's training portion.</p>
            </article>
            <article class="panel evidence-panel">
              <div class="panel-head"><div><p class="card-label">T+1 / T+5 · cents/lb MAE</p><h2>Feature groups</h2></div></div>
              @for (row of report.feature_ablation; track row.fold + '-' + row.horizon) {
                <div class="evidence-row"><strong>Fold {{ row.fold }} · T+{{ row.horizon }}</strong><span>Cotton {{ row.cotton_mae.toFixed(2) }}</span><span>+ macro {{ row.cotton_macro_mae.toFixed(2) }}</span><span>+ regime {{ row.full_mae.toFixed(2) }}</span></div>
              }
              <p class="evidence-footnote">CFTC excluded: actual publication timestamp is not verified. Full feature model results appear in the matrix.</p>
            </article>
          </div>
          @if (lstmHistory().length) {
            <article class="panel evidence-panel curve-panel"><div class="panel-head"><div><p class="card-label">Colab training trace</p><h2>LSTM learning curve</h2></div><span class="origin-chip">Train vs validation loss</span></div>
              <app-chart [options]="curveOptions()" label="LSTM train and validation loss across epochs" />
              <p class="evidence-footnote">{{ lstmParameters() }} · Early stopping restores the best validation epoch.</p>
            </article>
          }
        }
        <div class="definition-grid">
          <article><strong>MAE</strong><p>Average absolute price error in cents per pound. Primary selection metric.</p></article>
          <article><strong>RMSE</strong><p>Penalizes larger misses more heavily than MAE.</p></article>
          <article><strong>MAPE</strong><p>Error relative to realized price, expressed as a percentage.</p></article>
          <article><strong>Directional accuracy</strong><p>Share of forecasts with the correct return sign.</p></article>
        </div>
      }
    </section>
  `,
})
export class ModelLabPage {
  private readonly api = inject(ApiService);
  readonly metrics = signal<ModelMetric[]>([]);
  readonly evaluation = signal<ModelEvaluation | null>(null);
  readonly loading = signal(true);
  readonly error = signal('');
  readonly isFixture = computed(() => this.metrics().some((item) => item.data_quality === 'illustrative'));
  readonly lstmHistory = computed(() => this.metrics().find((item) => item.model === 'LSTM' && item.horizon === 1)?.training_history ?? []);
  readonly lstmParameters = computed(() => {
    const parameters = this.metrics().find((item) => item.model === 'LSTM' && item.horizon === 1)?.parameters;
    return parameters ? `${parameters['units']} units · ${parameters['dropout']} dropout · ${parameters['epochs_run']} epochs` : '';
  });
  readonly curveOptions = computed<EChartsCoreOption>(() => ({
    animationDuration: 300,
    grid: { left: 58, right: 20, top: 38, bottom: 38 },
    legend: { data: ['Train', 'Validation'], top: 2 },
    xAxis: { type: 'category', name: 'Epoch', data: this.lstmHistory().map((row) => row.epoch) },
    yAxis: { type: 'value', name: 'MAE loss', scale: true },
    tooltip: { trigger: 'axis' },
    series: [
      { name: 'Train', type: 'line', showSymbol: false, data: this.lstmHistory().map((row) => row.loss) },
      { name: 'Validation', type: 'line', showSymbol: false, data: this.lstmHistory().map((row) => row.val_loss) },
    ],
  }));

  constructor() {
    forkJoin({ metrics: this.api.metrics(), evaluation: this.api.evaluation() }).subscribe({
      next: ({ metrics, evaluation }) => { this.metrics.set(metrics); this.evaluation.set(evaluation); this.loading.set(false); },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Unknown API error'); this.loading.set(false); },
    });
  }

  selected(horizon: number): ModelMetric | undefined { return this.metrics().find((item) => item.horizon === horizon && item.selected); }
  decision(horizon: number): { locked_candidate?: string; selected?: string; fold_wins_vs_naive?: Record<string, number> } | null {
    return this.evaluation()?.selection_audit?.[String(horizon)] ?? null;
  }
  improvement(horizon: number): number {
    const naive = this.metrics().find((item) => item.horizon === horizon && item.model === 'Naive')?.walkforward?.mae;
    const candidate = this.selected(horizon)?.walkforward?.mae;
    return naive && candidate !== undefined ? (naive - candidate) / naive * 100 : 0;
  }
}
