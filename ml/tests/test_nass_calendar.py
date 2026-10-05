import json

import pytest
from cottonlens_ml.sources.nass_calendar import (
    CalendarHTML,
    archive_calendar_month,
    audit_condition_calendar,
)

HTML = ('<tr class="calendar"><td class="calendar">Mon, 06/05/23</td>'
        '<td class="calendar">3:00 pm ET</td><td class="calendar">Other</td>'
        '<td class="calendar">Published</td></tr>'
        '<tr class="calendar"><td class="calendar"></td>'
        '<td class="calendar">4:00 pm ET</td>'
        '<td class="calendar"><a href="./calendar-landing.php?source=n&amp;year=23&amp;month=06'
        '&amp;day=05&amp;report_id=17011">Crop Progress</a></td>'
        '<td class="calendar">Published</td></tr>')


class Response:
    status_code = 200

    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield self.raw


class Session:
    def get(self, url, **kwargs):
        assert kwargs['allow_redirects'] is False
        assert 'month=06&view=l&year=2023' in url
        return Response(HTML.encode())


def test_calendar_inherits_date_and_keeps_scheduled_time_distinct(tmp_path):
    folder, receipt = archive_calendar_month(2023, 6, tmp_path, session=Session())
    assert len(receipt['entries']) == 1
    entry = receipt['entries'][0]
    assert entry['release_date'] == '2023-06-05'
    assert entry['scheduled_time_et'] == '4:00 pm ET'
    assert entry['status_at_retrieval'] == 'Published'
    assert not receipt['actual_publication_clock_verified'] and not receipt['model_eligible']
    assert (folder / 'calendar.html').read_bytes() == HTML.encode()


def test_calendar_rejects_link_date_mismatch():
    parser = CalendarHTML(2023, 6)
    with pytest.raises(ValueError, match='disagrees'):
        parser.feed(HTML.replace('day=05', 'day=06'))


def test_audit_matches_condition_release_but_not_actual_clock(tmp_path):
    condition = tmp_path / 'condition.json'
    condition.write_text(json.dumps({'model_eligible': False, 'results': [
        {'status': 'numeric_match',
         'release_page': 'https://esmis.nal.usda.gov/publication/crop-progress/2023-06-05'},
    ]}), encoding='utf-8')
    result = audit_condition_calendar(condition, tmp_path / 'archive', session=Session())
    assert result['matched'] == result['total'] == 1
    assert result['results'][0]['scheduled_time_et'] == '4:00 pm ET'
    assert not result['actual_publication_clock_verified'] and not result['model_eligible']


@pytest.mark.parametrize('replacement', ['3:00 pm ET', 'Revised'])
def test_audit_does_not_count_unverified_schedule_or_status(tmp_path, replacement):
    condition = tmp_path / 'condition.json'
    condition.write_text(json.dumps({'model_eligible': False, 'results': [
        {'status': 'numeric_match',
         'release_page': 'https://esmis.nal.usda.gov/publication/crop-progress/2023-06-05'},
    ]}), encoding='utf-8')

    class ChangedSession(Session):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            response.raw = HTML.replace('4:00 pm ET' if 'pm' in replacement else 'Published',
                                        replacement).encode()
            return response

    result = audit_condition_calendar(condition, tmp_path / 'archive', session=ChangedSession())
    assert result['matched'] == 0
    assert result['results'][0]['status'] == 'calendar_unverified'
    assert not result['model_eligible']
