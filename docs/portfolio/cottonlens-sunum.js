// research/evidence/momentum-full-2002-2025-20261116.json, USDA then revision rule.
const reports = [
  { month: 'Eylül', US: [6.3261, 5.9870], TX: [12.9043, 11.9435], state_w: [10.9717, 10.4581] },
  { month: 'Ekim', US: [5.4818, 4.8727], TX: [12.0818, 11.2727], state_w: [9.3107, 9.0634] },
  { month: 'Kasım', US: [3.5500, 3.4625], TX: [8.0417, 7.4583], state_w: [6.4078, 6.2324] },
  { month: 'Aralık', US: [2.4875, 2.2917], TX: [5.2542, 4.9833], state_w: [4.4620, 4.3439] },
];

// research/evidence/t49-anchoring-20261116.json, Texas S0.
const texasAreaReports = [
  { month: 'Eylül', area: [12.8128, 7.8611] },
  { month: 'Ekim', area: [13.1657, 7.6243] },
  { month: 'Kasım', area: [12.6355, 7.0383] },
  { month: 'Aralık', area: [12.6355, 7.3017] },
];

const regionControls = document.querySelector('.region-control');
const chart = document.querySelector('#error-chart');
const regionalTrend = document.querySelector('#regional-trend');
const heroTrend = document.querySelector('#hero-trend');
const percent = new Intl.NumberFormat('tr-TR', {
  style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2,
});
const series = [
  { name: 'USDA', color: 'reference' },
  { name: 'CottonLensAI', color: 'corrected' },
];
let activeRegion = 'US';

function plotTrend(canvas, region, maximum, rows = reports) {
  const { width, height } = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const context = canvas.getContext('2d');
  const styles = getComputedStyle(canvas);
  const left = 37;
  const top = 13;
  const bottom = height - 25;
  const right = width - 16;
  const x = index => left + (right - left) * index / (rows.length - 1);
  const y = value => bottom - (bottom - top) * value / maximum;

  context.scale(scale, scale);
  context.font = '12px "DM Sans", sans-serif';
  context.textBaseline = 'middle';
  context.strokeStyle = styles.getPropertyValue('--line').trim();
  context.fillStyle = styles.getPropertyValue('--muted').trim();
  for (let tick = 0; tick <= 3; tick += 1) {
    const value = maximum * tick / 3;
    context.textAlign = 'left';
    context.fillText(`%${value}`, 0, y(value));
    context.beginPath();
    context.moveTo(left, y(value));
    context.lineTo(right, y(value));
    context.stroke();
  }
  rows.forEach(({ month }, index) => {
    context.textAlign = index === 0 ? 'left' : index === rows.length - 1 ? 'right' : 'center';
    context.fillText(month, x(index), height - 8);
  });

  for (const [index, { color }] of series.entries()) {
    const coordinates = rows.map((report, month) => [x(month), y(report[region][index])]);
    const lineColor = styles.getPropertyValue(`--${color}`).trim();
    context.strokeStyle = lineColor;
    context.fillStyle = lineColor;
    context.lineWidth = 2.5;
    context.beginPath();
    context.moveTo(...coordinates[0]);
    coordinates.slice(1).forEach(point => context.lineTo(...point));
    context.stroke();
    coordinates.forEach(point => {
      context.beginPath();
      context.arc(...point, 3, 0, 2 * Math.PI);
      context.fill();
    });
  }
}

function drawErrors(region, regionName) {
  activeRegion = region;
  chart.innerHTML = `
    <table class="error-table">
      <caption class="sr-only">${regionName} · Aylık tahmin hatası (MAPE)</caption>
      <thead><tr><th scope="col">Rapor ayı</th>${series.map(({ name }) => `<th scope="col">${name}</th>`).join('')}</tr></thead>
      <tbody>
        ${reports.map(({ month, [region]: values }) => `
          <tr><th scope="row">${month}</th>${values.map(value => `<td class="series-value">${percent.format(value / 100)}</td>`).join('')}</tr>
        `).join('')}
      </tbody>
    </table>`;
  plotTrend(regionalTrend, activeRegion, 15);
}

regionControls.addEventListener('click', ({ target }) => {
  const selected = target.closest('button[data-region]');
  if (!selected || selected.getAttribute('aria-pressed') === 'true') return;

  regionControls.querySelector('[aria-pressed="true"]').setAttribute('aria-pressed', 'false');
  selected.setAttribute('aria-pressed', 'true');
  drawErrors(selected.dataset.region, selected.textContent);
});

drawErrors('US', 'ABD');

const trendResize = new ResizeObserver(entries => {
  for (const { target } of entries) {
    if (target === heroTrend) plotTrend(target, 'area', 15, texasAreaReports);
    else plotTrend(target, activeRegion, 15);
  }
});
trendResize.observe(heroTrend);
trendResize.observe(regionalTrend);

document.fonts.ready.then(() => {
  plotTrend(heroTrend, 'area', 15, texasAreaReports);
  plotTrend(regionalTrend, activeRegion, 15);
});

const satelliteTrack = document.querySelector('.satellite-track');
const satelliteGroup = satelliteTrack.firstElementChild;

function repeatSatelliteTiles() {
  const tiles = [...satelliteGroup.children];
  for (const tile of tiles) {
    const copy = tile.cloneNode(true);
    copy.setAttribute('aria-hidden', 'true');
    satelliteGroup.append(copy);
  }
  const duplicate = satelliteGroup.cloneNode(true);
  duplicate.setAttribute('aria-hidden', 'true');
  satelliteTrack.append(duplicate);
  satelliteTrack.parentElement.classList.add('is-looping');
}

repeatSatelliteTiles();
