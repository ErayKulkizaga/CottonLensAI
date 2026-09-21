import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../api.service';
import { LatestForecast, SimulationAdjustments, SimulationResponse } from '../types';

@Component({
  standalone: true,
  imports: [FormsModule],
  template: `
    <section class="page">
      <header class="page-head">
        <div><p class="eyebrow">Counterfactual model response</p><h1>Market sensitivity lab</h1></div>
        <div class="disclaimer-chip">Sensitivity simulation · not causal inference</div>
      </header>
      @if (loadingBase()) { <div class="state-panel"><span class="loader"></span><p>Preparing the baseline snapshot…</p></div> }
      @else if (error()) { <div class="state-panel error"><strong>Simulation unavailable</strong><p>{{ error() }}</p></div> }
      @else if (latest(); as data) {
        @if (data.data_quality === 'illustrative') {
          <div class="fixture-banner"><strong>Development fixture</strong><span>The control surface is functional, but values are not trained market evidence.</span></div>
        }
        <div class="lab-layout">
          <form class="panel control-panel" (ngSubmit)="run()">
            <div class="panel-head"><div><p class="card-label">Scenario controls</p><h2>Shift the market state</h2></div><button class="text-button" type="button" (click)="reset()">Reset all</button></div>
            <label class="range-control">
              <span><strong>Dollar index</strong><output>{{ signed(adjustments.dxy_pct_change) }}%</output></span>
              <input type="range" min="-5" max="5" step="0.25" [(ngModel)]="adjustments.dxy_pct_change" name="dxy" />
              <small><span>−5%</span><span>Baseline</span><span>+5%</span></small>
            </label>
            <label class="range-control">
              <span><strong>WTI crude</strong><output>{{ signed(adjustments.wti_pct_change) }}%</output></span>
              <input type="range" min="-20" max="20" step="1" [(ngModel)]="adjustments.wti_pct_change" name="wti" />
              <small><span>−20%</span><span>Baseline</span><span>+20%</span></small>
            </label>
            <label class="range-control">
              <span><strong>CFTC managed money</strong><output>{{ signedInteger(adjustments.cftc_net_delta_contracts) }}</output></span>
              <input type="range" min="-50000" max="50000" step="2500" [(ngModel)]="adjustments.cftc_net_delta_contracts" name="cftc" />
              <small><span>−50k</span><span>Contracts</span><span>+50k</span></small>
            </label>
            <label class="range-control">
              <span><strong>20-day volatility</strong><output>{{ adjustments.volatility_multiplier.toFixed(2) }}×</output></span>
              <input type="range" min="0.5" max="2" step="0.05" [(ngModel)]="adjustments.volatility_multiplier" name="volatility" />
              <small><span>0.5×</span><span>Baseline</span><span>2.0×</span></small>
            </label>
            <button class="primary-button" type="submit" [disabled]="running()">
              @if (running()) { <span class="button-loader"></span>Running inference… } @else { Run simulation <span>→</span> }
            </button>
          </form>
          <div class="result-column">
            <article class="scenario-hero">
              <p class="card-label">Scenario response</p>
              @if (result(); as simulation) {
                <div class="scenario-results">
                  @for (item of simulation.results; track item.horizon) {
                    <div class="scenario-result">
                      <span>T+{{ item.horizon }}</span>
                      <div><small>Baseline</small><strong>{{ item.baseline_price_cents_per_lb.toFixed(2) }}¢</strong></div>
                      <div class="bridge" [class.negative]="item.delta_pct < 0"><i></i><span>{{ signed(item.delta_pct) }}%</span></div>
                      <div><small>Scenario</small><strong>{{ item.scenario_price_cents_per_lb.toFixed(2) }}¢</strong></div>
                    </div>
                  }
                </div>
                <p class="run-id">Run {{ simulation.id.slice(0, 8) }} · {{ simulation.model_version }}</p>
              } @else {
                <div class="empty-result"><span class="cotton-orbit">◎</span><strong>Baseline is ready</strong><p>Change one or more market inputs, then run inference to see the model response.</p></div>
              }
            </article>
            <article class="method-card"><strong>What this answers</strong><p>“How does the fitted model respond if selected inputs change while every other feature stays fixed?”</p><strong>What it does not answer</strong><p>It does not estimate a causal market effect or account for feedback between variables.</p></article>
          </div>
        </div>
      }
    </section>
  `,
})
export class SensitivityPage {
  private readonly api = inject(ApiService);
  readonly latest = signal<LatestForecast | null>(null);
  readonly result = signal<SimulationResponse | null>(null);
  readonly loadingBase = signal(true);
  readonly running = signal(false);
  readonly error = signal('');
  adjustments: SimulationAdjustments = this.defaults();

  constructor() {
    this.api.latest().subscribe({
      next: (value) => { this.latest.set(value); this.loadingBase.set(false); },
      error: (error: { message?: string }) => { this.error.set(error.message ?? 'Unknown API error'); this.loadingBase.set(false); },
    });
  }

  run(): void {
    const latest = this.latest();
    if (!latest) return;
    this.running.set(true); this.error.set('');
    this.api.simulate(latest.data_as_of, this.adjustments).subscribe({
      next: (value) => { this.result.set(value); this.running.set(false); },
      error: (error: { error?: { message?: string }; message?: string }) => { this.error.set(error.error?.message ?? error.message ?? 'Simulation failed'); this.running.set(false); },
    });
  }

  reset(): void { this.adjustments = this.defaults(); this.result.set(null); }
  signed(value: number): string { return `${value >= 0 ? '+' : ''}${Number(value).toFixed(2)}`; }
  signedInteger(value: number): string { return `${value >= 0 ? '+' : ''}${Number(value).toLocaleString()}`; }
  private defaults(): SimulationAdjustments { return { dxy_pct_change: 0, wti_pct_change: 0, cftc_net_delta_contracts: 0, volatility_multiplier: 1 }; }
}

