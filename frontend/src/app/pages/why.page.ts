import { Component, computed, inject, signal } from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';
import { switchMap } from 'rxjs';

import { ApiService } from '../api.service';
import { ChartComponent } from '../chart.component';
import { Explanation, LatestForecast } from '../types';

@Component({
  standalone: true,
  imports: [ChartComponent],
  template: `
    <section class="page">
      <header class="page-head"><div><p class="eyebrow">Local explanation · T+1</p><h1>Why this forecast?</h1></div></header>
      @if (loading()) { <div class="state-panel"><span class="loader"></span><p>Reconstructing feature contributions…</p></div> }
      @else if (error()) { <div class="state-panel error"><strong>Explanation unavailable</strong><p>{{ error() }}</p></div> }
      @else if (explanation(); as exp) {
        @if (latest()?.data_quality === 'illustrative') {
          <div class="fixture-banner"><strong>Development fixture</strong><span>Contribution values below are illustrative and not trained market evidence.</span></div>
        }
        <div class="why-layout">
          <article class="panel contribution-panel">
            <div class="panel-head"><div><p class="card-label">Prediction anatomy</p><h2>Feature contributions</h2></div><span class="model-tag">{{ exp.explainer }}</span></div>
            <app-chart [options]="contributionOptions()" label="Positive and negative feature contributions" />
          </article>
          <aside class="equation-card">
            <p class="card-label">Additive explanation</p>
            <div class="equation">ŷ = E[f(X)] + Σ φ<sub>i</sub></div>
            <dl>
              <div><dt>ŷ</dt><dd>Predicted return: <strong>{{ signed(exp.predicted_return_pct) }}%</strong></dd></div>
              <div><dt>E[f(X)]</dt><dd>Model baseline: {{ signed(exp.base_value_pct) }}%</dd></div>
              <div><dt>φ<sub>i</sub></dt><dd>Each feature’s local contribution</dd></div>
              @if (Math.abs(exp.approximation_error_pct ?? 0) >= 0.01) { <div><dt>ε</dt><dd>Approximation residual: {{ signed(exp.approximation_error_pct ?? 0) }} pp</dd></div> }
            </dl>
            <div class="decision-note"><strong>Reading rule</strong><p>Rightward bars support the forecast; leftward bars oppose it. Contribution is association inside the fitted model, not causality.</p></div>
          </aside>
        </div>
        <article class="panel feature-ledger">
          <div class="panel-head"><div><p class="card-label">Evidence ledger</p><h2>Top drivers</h2></div><span class="origin-chip">As of {{ latest()?.data_as_of }}</span></div>
          <div class="feature-table">
            @for (item of sortedContributions(); track item.feature) {
              <div class="feature-row">
                <span><strong>{{ item.display_name }}</strong><small>{{ item.feature }}</small></span>
                <span class="data-value">{{ item.feature === 'other' ? '—' : item.feature_value.toFixed(3) }}</span>
                <span [class.positive]="item.contribution_pct >= 0" [class.negative]="item.contribution_pct < 0">{{ signed(item.contribution_pct) }} pp</span>
              </div>
            }
          </div>
        </article>
      }
    </section>
  `,
})
export class WhyPage {
  readonly Math = Math;
  private readonly api = inject(ApiService);
  readonly latest = signal<LatestForecast | null>(null);
  readonly explanation = signal<Explanation | null>(null);
  readonly loading = signal(true);
  readonly error = signal('');
  readonly sortedContributions = computed(() => [...(this.explanation()?.contributions ?? [])].sort((a, b) => Math.abs(b.contribution_pct) - Math.abs(a.contribution_pct)));
  readonly contributionOptions = computed<EChartsCoreOption>(() => {
    const items = [...this.sortedContributions()].reverse();
    return {
      animationDuration: 500,
      grid: { left: 158, right: 30, top: 18, bottom: 30 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#102b2a', borderWidth: 0, textStyle: { color: '#f5f0e6' } },
      xAxis: { type: 'value', axisLabel: { formatter: '{value} pp', color: '#78827c' }, splitLine: { lineStyle: { color: '#ebe8de' } } },
      yAxis: { type: 'category', data: items.map((item) => item.display_name), axisLabel: { color: '#37433e', width: 135, overflow: 'truncate' }, axisLine: { show: false }, axisTick: { show: false } },
      series: [{ type: 'bar', data: items.map((item) => ({ value: item.contribution_pct, itemStyle: { color: item.contribution_pct >= 0 ? '#2f7468' : '#a34b3f', borderRadius: item.contribution_pct >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] } })), barWidth: 18 }],
    };
  });

  constructor() {
    this.api.latest().pipe(switchMap((latest) => { this.latest.set(latest); return this.api.explanation(latest.forecasts.find((item) => item.horizon === 1)!.id); })).subscribe({
      next: (value) => { this.explanation.set(value); this.loading.set(false); },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Unknown API error'); this.loading.set(false); },
    });
  }

  signed(value: number): string { return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`; }
}
