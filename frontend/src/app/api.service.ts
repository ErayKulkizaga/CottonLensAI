import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import {
  Explanation,
  Forecast,
  LatestForecast,
  MarketHistory,
  ModelMetric,
  ModelEvaluation,
  ReplayResponse,
  SimulationAdjustments,
  SimulationResponse,
} from './types';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = '/api/v1';

  latest(): Observable<LatestForecast> {
    return this.http.get<LatestForecast>(`${this.base}/forecasts/latest`);
  }

  history(horizon: 1 | 5 = 1, limit = 80): Observable<Forecast[]> {
    const params = new HttpParams()
      .set('horizon', horizon)
      .set('origin_type', 'backtest')
      .set('limit', limit);
    return this.http.get<Forecast[]>(`${this.base}/forecasts/history`, { params });
  }

  market(): Observable<MarketHistory> {
    return this.http.get<MarketHistory>(`${this.base}/market/history`, {
      params: new HttpParams().append('series', 'cotton'),
    });
  }

  explanation(forecastId: string): Observable<Explanation> {
    return this.http.get<Explanation>(`${this.base}/forecasts/${forecastId}/explanation`);
  }

  metrics(): Observable<ModelMetric[]> {
    return this.http.get<ModelMetric[]>(`${this.base}/models/metrics`);
  }

  evaluation(): Observable<ModelEvaluation> {
    return this.http.get<ModelEvaluation>(`${this.base}/models/evaluation`);
  }

  replay(date: string): Observable<ReplayResponse> {
    return this.http.get<ReplayResponse>(`${this.base}/replay/${date}`);
  }

  simulate(asOfDate: string, adjustments: SimulationAdjustments): Observable<SimulationResponse> {
    return this.http.post<SimulationResponse>(`${this.base}/simulations`, {
      as_of_date: asOfDate,
      adjustments,
    });
  }
}
