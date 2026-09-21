import { Component, computed, inject, signal } from '@angular/core';

import { ApiService } from '../api.service';
import { ModelMetric } from '../types';

@Component({
  standalone: true,
  template: `
    <section class="page">
      <header class="page-head"><div><p class="eyebrow">Locked holdout comparison</p><h1>Model lab</h1></div><div class="policy-note">LSTM needs ≥5% lower MAE and no directional loss to displace XGBoost.</div></header>
      @if (loading()) { <div class="state-panel"><span class="loader"></span><p>Loading experiment evidence…</p></div> }
      @else if (error()) { <div class="state-panel error"><strong>Metrics unavailable</strong><p>{{ error() }}</p></div> }
      @else {
        @if (isFixture()) { <div class="fixture-banner"><strong>Development fixture</strong><span>Metric values are illustrative placeholders, never production claims.</span></div> }
        <div class="model-summary">
          @for (horizon of [1, 5]; track horizon) {
            <article class="winner-card">
              <span>T+{{ horizon }} production</span>
              <strong>{{ selected(horizon)?.model }}</strong>
              <p>MAE {{ selected(horizon)?.mae?.toFixed(2) }} ¢/lb</p>
            </article>
          }
          <article class="policy-card"><span>Selection guardrail</span><strong>Accuracy earns entry. Explainability breaks close ties.</strong></article>
        </div>
        <article class="panel metric-panel">
          <div class="panel-head"><div><p class="card-label">Comparable test window</p><h2>Performance matrix</h2></div><span class="origin-chip">Lower error is better</span></div>
          <div class="metric-table" role="table" aria-label="Model metrics">
            <div class="metric-row metric-header" role="row"><span>Model</span><span>Horizon</span><span>MAE</span><span>RMSE</span><span>MAPE</span><span>Direction</span></div>
            @for (metric of metrics(); track metric.model + metric.horizon) {
              <div class="metric-row" [class.selected]="metric.selected" role="row">
                <span><strong>{{ metric.model }}</strong>@if (metric.selected) { <small>Selected</small> }</span>
                <span>T+{{ metric.horizon }}</span>
                <span>{{ metric.mae.toFixed(2) }}</span>
                <span>{{ metric.rmse.toFixed(2) }}</span>
                <span>{{ metric.mape.toFixed(2) }}%</span>
                <span>{{ metric.directional_accuracy.toFixed(1) }}%</span>
              </div>
            }
          </div>
        </article>
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
  readonly loading = signal(true);
  readonly error = signal('');
  readonly isFixture = computed(() => this.metrics().some((item) => item.data_quality === 'illustrative'));

  constructor() {
    this.api.metrics().subscribe({
      next: (value) => { this.metrics.set(value); this.loading.set(false); },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Unknown API error'); this.loading.set(false); },
    });
  }

  selected(horizon: number): ModelMetric | undefined { return this.metrics().find((item) => item.horizon === horizon && item.selected); }
}

