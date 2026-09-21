import { Component, computed, inject, signal } from '@angular/core';
import { forkJoin } from 'rxjs';
import type { EChartsCoreOption } from 'echarts/core';

import { ApiService } from '../api.service';
import { ChartComponent } from '../chart.component';
import { Forecast, LatestForecast, MarketHistory } from '../types';

@Component({
  standalone: true,
  imports: [ChartComponent],
  template: `
    <section class="page page-dashboard">
      <header class="page-head">
        <div><p class="eyebrow">ICE Cotton No. 2 · Short-horizon view</p><h1>Forecast desk</h1></div>
        @if (latest(); as data) {
          <div class="meta-strip">
            <span><small>Data as of</small>{{ data.data_as_of }}</span>
            <span><small>Artifact</small>{{ data.artifact_version }}</span>
            <span class="quality" [class.fixture]="data.data_quality === 'illustrative'">
              <small>Evidence</small>{{ qualityLabel(data.data_quality) }}
            </span>
          </div>
        }
      </header>

      @if (loading()) {
        <div class="state-panel"><span class="loader"></span><p>Loading the forecast surface…</p></div>
      } @else if (error()) {
        <div class="state-panel error"><strong>Forecast surface unavailable</strong><p>{{ error() }}</p><button (click)="load()">Try again</button></div>
      } @else if (latest(); as data) {
        @if (data.data_quality === 'illustrative') {
          <div class="fixture-banner"><strong>Development fixture</strong><span>These values exercise the product flow; they are not trained market results. Import a validated Colab artifact to replace them.</span></div>
        }
        @if (data.stale) {
          <div class="stale-banner">The active artifact is more than five days old. Rerun the Colab pipeline before making a current-market claim.</div>
        }
        <div class="forecast-stage">
          <article class="spot-card">
            <p class="card-label">Current cotton proxy</p>
            <strong class="spot-value">{{ data.forecasts[0].current_price_cents_per_lb.toFixed(2) }}</strong>
            <span class="unit">¢ / lb</span>
            <div class="spot-rule"></div>
            <p>Continuous futures proxy</p>
            <small>Not an official ICE settlement feed</small>
          </article>
          <div class="forecast-pair">
            @for (forecast of data.forecasts; track forecast.id) {
              <article class="forecast-card" [class.down]="forecast.direction === 'down'">
                <div class="forecast-top"><span>T+{{ forecast.horizon }}</span><span class="model-tag">{{ forecast.model_name }}</span></div>
                <p>Expected close</p>
                <strong>{{ forecast.predicted_price_cents_per_lb.toFixed(2) }}<small>¢</small></strong>
                <div class="movement"><span>{{ forecast.direction === 'up' ? '↗' : forecast.direction === 'down' ? '↘' : '→' }}</span>{{ signed(forecast.predicted_return_pct) }}%</div>
                <small>Target {{ forecast.target_date }}</small>
              </article>
            }
          </div>
        </div>

        <div class="dashboard-grid">
          <article class="panel chart-panel">
            <div class="panel-head"><div><p class="card-label">Market tape</p><h2>Cotton price history</h2></div><span class="legend-line">Close</span></div>
            <app-chart [options]="priceOptions()" label="Historical cotton close prices" />
          </article>
          <article class="panel chart-panel">
            <div class="panel-head"><div><p class="card-label">Holdout evidence</p><h2>Backtest: forecast vs actual</h2></div><span class="origin-chip">Backtest</span></div>
            <app-chart [options]="backtestOptions()" label="Backtest forecast and actual prices" />
          </article>
        </div>
        <p class="source-note">{{ market()?.source_note }}</p>
      }
    </section>
  `,
})
export class DashboardPage {
  private readonly api = inject(ApiService);
  readonly latest = signal<LatestForecast | null>(null);
  readonly market = signal<MarketHistory | null>(null);
  readonly history = signal<Forecast[]>([]);
  readonly loading = signal(true);
  readonly error = signal('');

  readonly priceOptions = computed<EChartsCoreOption>(() => {
    const points = this.market()?.points ?? [];
    return baseLineOptions(
      points.map((point) => point.date),
      [{ name: 'Cotton close', data: points.map((point) => point.close), color: '#bd7436' }],
    );
  });

  readonly backtestOptions = computed<EChartsCoreOption>(() => {
    const points = this.history();
    return baseLineOptions(
      points.map((point) => point.target_date),
      [
        { name: 'Forecast', data: points.map((point) => point.predicted_price_cents_per_lb), color: '#d29b56', lineType: 'dashed' },
        { name: 'Actual', data: points.map((point) => point.actual_price_cents_per_lb), color: '#204f4b', lineType: 'solid' },
      ],
    );
  });

  constructor() { this.load(); }

  load(): void {
    this.loading.set(true);
    this.error.set('');
    forkJoin({ latest: this.api.latest(), market: this.api.market(), history: this.api.history(1, 80) }).subscribe({
      next: ({ latest, market, history }) => {
        this.latest.set(latest); this.market.set(market); this.history.set(history); this.loading.set(false);
      },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Unknown API error'); this.loading.set(false); },
    });
  }

  signed(value: number): string { return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`; }
  qualityLabel(value: string): string { return value === 'illustrative' ? 'Fixture only' : 'Validated holdout'; }
}

function baseLineOptions(
  dates: string[],
  series: Array<{ name: string; data: Array<number | null>; color: string; lineType?: 'solid' | 'dashed' }>,
): EChartsCoreOption {
  return {
    animationDuration: 550,
    color: series.map((item) => item.color),
    grid: { left: 48, right: 20, top: 38, bottom: 38 },
    tooltip: { trigger: 'axis', backgroundColor: '#102b2a', borderWidth: 0, textStyle: { color: '#f5f0e6' } },
    legend: { show: series.length > 1, top: 0, right: 0, textStyle: { color: '#68736d' } },
    xAxis: { type: 'category', data: dates, axisLabel: { color: '#78827c', hideOverlap: true }, axisLine: { lineStyle: { color: '#d8d5ca' } } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#78827c' }, splitLine: { lineStyle: { color: '#ebe8de' } } },
    series: series.map((item) => ({ name: item.name, type: 'line', data: item.data, showSymbol: false, smooth: 0.2, lineStyle: { width: 2, type: item.lineType ?? 'solid' } })),
  };
}
