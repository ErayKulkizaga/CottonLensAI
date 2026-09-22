import { Component, inject, signal } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { ApiService } from './api.service';
import { LatestForecast } from './types';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, RouterOutlet],
  template: `
    <div class="app-shell">
      <aside class="rail">
        <a class="brand" routerLink="/" aria-label="CottonLens home">
          <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
          <span><strong>CottonLens</strong><small>Market intelligence</small></span>
        </a>
        <nav aria-label="Primary navigation">
          <a routerLink="/" [routerLinkActiveOptions]="{ exact: true }" routerLinkActive="active">
            <span class="nav-index">01</span><span>Forecast</span>
          </a>
          <a routerLink="/why" routerLinkActive="active"><span class="nav-index">02</span><span>Why?</span></a>
          <a routerLink="/sensitivity" routerLinkActive="active"><span class="nav-index">03</span><span>Sensitivity</span></a>
          <a routerLink="/models" routerLinkActive="active"><span class="nav-index">04</span><span>Model Lab</span></a>
          <a routerLink="/replay" routerLinkActive="active"><span class="nav-index">05</span><span>Replay</span></a>
        </nav>
        <div class="rail-foot">
          <span class="pulse" aria-hidden="true"></span>
          <span>Local inference</span>
          <small>Training: Google Colab</small>
        </div>
      </aside>
      <header class="mobile-head">
        <a class="brand" routerLink="/"><span class="brand-mark"><i></i><i></i><i></i></span><strong>CottonLens</strong></a>
        <nav aria-label="Mobile navigation">
          <a routerLink="/" [routerLinkActiveOptions]="{ exact: true }" routerLinkActive="active">Forecast</a>
          <a routerLink="/why" routerLinkActive="active">Why</a>
          <a routerLink="/sensitivity" routerLinkActive="active">Simulate</a>
          <a routerLink="/models" routerLinkActive="active">Models</a>
          <a routerLink="/replay" routerLinkActive="active">Replay</a>
        </nav>
      </header>
      <main class="workspace">
        @if (router.url !== '/' && latest(); as data) {
          <div class="workspace-meta" aria-label="Current artifact context">
            <span>Data {{ data.data_as_of }}</span><span>Artifact {{ data.artifact_version }}</span>
            <span>{{ data.forecasts[0].model_name }} T+1 · {{ data.forecasts[1].model_name }} T+5</span>
            <span>{{ data.data_quality === 'illustrative' ? 'Illustrative fixture' : (data.stale ? 'Stale data' : 'Latest available data') }} · live snapshot</span>
          </div>
        }
        <router-outlet />
      </main>
    </div>
  `,
})
export class AppComponent {
  readonly router = inject(Router);
  private readonly api = inject(ApiService);
  readonly latest = signal<LatestForecast | null>(null);

  constructor() {
    this.api.latest().subscribe({ next: (data) => this.latest.set(data) });
  }
}
