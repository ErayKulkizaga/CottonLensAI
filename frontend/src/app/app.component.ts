import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

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
      <main class="workspace"><router-outlet /></main>
    </div>
  `,
})
export class AppComponent {}

