import { Routes } from '@angular/router';

import { DashboardPage } from './pages/dashboard.page';
import { WhyPage } from './pages/why.page';
import { SensitivityPage } from './pages/sensitivity.page';
import { ModelLabPage } from './pages/model-lab.page';
import { ReplayPage } from './pages/replay.page';

export const routes: Routes = [
  { path: '', component: DashboardPage, title: 'Forecast · CottonLens AI' },
  { path: 'why', component: WhyPage, title: 'Why? · CottonLens AI' },
  { path: 'sensitivity', component: SensitivityPage, title: 'Sensitivity Lab · CottonLens AI' },
  { path: 'models', component: ModelLabPage, title: 'Model Lab · CottonLens AI' },
  { path: 'replay', component: ReplayPage, title: 'Prediction Replay · CottonLens AI' },
  { path: '**', redirectTo: '' },
];

