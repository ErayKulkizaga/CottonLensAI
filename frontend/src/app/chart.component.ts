import {
  AfterViewInit,
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import * as echarts from 'echarts/core';
import { LineChart, BarChart } from 'echarts/charts';
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { ECharts, EChartsCoreOption } from 'echarts/core';

echarts.use([LineChart, BarChart, GridComponent, LegendComponent, TooltipComponent, DataZoomComponent, CanvasRenderer]);

@Component({
  selector: 'app-chart',
  standalone: true,
  template: '<div #host class="chart-host" role="img" [attr.aria-label]="label"></div>',
  styles: '.chart-host{width:100%;height:100%;min-height:260px;min-width:0}',
})
export class ChartComponent implements AfterViewInit, OnChanges, OnDestroy {
  @ViewChild('host', { static: true }) host!: ElementRef<HTMLDivElement>;
  @Input({ required: true }) options!: EChartsCoreOption;
  @Input() label = 'Data chart';

  private chart?: ECharts;
  private observer?: ResizeObserver;

  ngAfterViewInit(): void {
    this.chart = echarts.init(this.host.nativeElement, undefined, { renderer: 'canvas' });
    this.chart.setOption(this.options);
    this.observer = new ResizeObserver(() => this.chart?.resize());
    this.observer.observe(this.host.nativeElement);
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['options'] && this.chart) this.chart.setOption(this.options, true);
  }

  ngOnDestroy(): void {
    this.observer?.disconnect();
    this.chart?.dispose();
  }
}

