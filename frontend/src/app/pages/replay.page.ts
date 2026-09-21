import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../api.service';
import { Forecast, ReplayResponse } from '../types';

@Component({
  standalone: true,
  imports: [FormsModule],
  template: `
    <section class="page">
      <header class="page-head">
        <div><p class="eyebrow">Point-in-time audit</p><h1>Prediction replay</h1></div>
        <label class="date-control"><span>Forecast date</span><select [(ngModel)]="selectedDate" (ngModelChange)="loadReplay($event)">@for (date of dates(); track date) { <option [value]="date">{{ date }}</option> }</select></label>
      </header>
      <div class="backtest-banner"><strong>Historical backtest</strong><span>These predictions were reconstructed using locked model inputs—not recorded live forecasts.</span></div>
      @if (loading()) { <div class="state-panel"><span class="loader"></span><p>Replaying the selected market state…</p></div> }
      @else if (error()) { <div class="state-panel error"><strong>Replay unavailable</strong><p>{{ error() }}</p></div> }
      @else if (replay(); as data) {
        <div class="replay-grid">
          @for (forecast of data.forecasts; track forecast.id) {
            <article class="replay-card">
              <div class="replay-head"><span>T+{{ forecast.horizon }}</span><span [class.correct]="forecast.direction_correct" [class.incorrect]="!forecast.direction_correct">{{ forecast.direction_correct ? 'Direction correct' : 'Direction missed' }}</span></div>
              <div class="price-journey">
                <span><small>Known close</small><strong>{{ forecast.current_price_cents_per_lb.toFixed(2) }}¢</strong></span>
                <i aria-hidden="true">→</i>
                <span><small>Forecast</small><strong>{{ forecast.predicted_price_cents_per_lb.toFixed(2) }}¢</strong></span>
                <i aria-hidden="true">→</i>
                <span><small>Actual</small><strong>{{ forecast.actual_price_cents_per_lb?.toFixed(2) }}¢</strong></span>
              </div>
              <div class="error-strip"><span>Absolute error</span><strong>{{ forecast.absolute_error?.toFixed(2) }} ¢/lb</strong></div>
              <footer><span>{{ forecast.model_name }}</span><span>Target {{ forecast.target_date }}</span></footer>
            </article>
          }
        </div>
        <article class="panel replay-note"><p class="card-label">Audit note</p><h2>What was knowable on {{ data.as_of_date }}?</h2><p>The replay uses only features whose availability date was on or before the forecast origin. Weekly CFTC positioning is delayed until its Friday publication window.</p></article>
      }
    </section>
  `,
})
export class ReplayPage {
  private readonly api = inject(ApiService);
  readonly dates = signal<string[]>([]);
  readonly replay = signal<ReplayResponse | null>(null);
  readonly loading = signal(true);
  readonly error = signal('');
  selectedDate = '';

  constructor() {
    this.api.history(1, 120).subscribe({
      next: (rows: Forecast[]) => {
        const dates = [...new Set(rows.map((row) => row.as_of_date))].reverse();
        this.dates.set(dates); this.selectedDate = dates[0] ?? '';
        if (this.selectedDate) this.loadReplay(this.selectedDate); else this.loading.set(false);
      },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Unknown API error'); this.loading.set(false); },
    });
  }

  loadReplay(date: string): void {
    this.loading.set(true); this.error.set('');
    this.api.replay(date).subscribe({
      next: (value) => { this.replay.set(value); this.loading.set(false); },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Replay failed'); this.loading.set(false); },
    });
  }
}

