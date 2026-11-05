import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('risk_readiness', Path(__file__).parents[1]/'risk_readiness.py')
risk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk)


def quote(day, first='Mar-23', second='May-23', publication=None):
    return {'report_date':day,'publication_local_naive':publication,'document_sha256':'a'*64,
            'first_contract':first,'second_contract':second,'quotes':{first:80.,second:81.}}


class RiskReadinessTests(unittest.TestCase):
    def test_square_requires_interaction(self):
        p=risk.components(80,88,80)
        self.assertAlmostEqual(p['total_squared'],0)
        self.assertGreater(p['gap_squared']+p['within_bar_squared'],0)
        self.assertLess(p['interaction'],0)
        self.assertAlmostEqual(sum(p[k] for k in ['gap_squared','within_bar_squared','interaction']),0)

    def test_invalid_component_prices_rejected(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):risk.components(80,value,81)

    def test_origin_contract_stays_fixed_when_future_front_changes(self):
        current=quote('2023-01-03')
        future=quote('2023-01-04',first='May-23',second='Jul-23')
        future['quotes']['Mar-23']=79.
        label=risk.named_label(current,future,'2023-01-03','2023-01-04','first')
        self.assertTrue(label['eligible'])
        self.assertEqual(label['contract'],'Mar-23')
        self.assertAlmostEqual(label['risk'],math.log(79/80)**2)
        self.assertFalse(label['historical_availability_verified'])

    def test_missing_contract_not_replaced_with_future_front(self):
        label=risk.named_label(quote('2023-01-03'),quote('2023-01-04','May-23','Jul-23'),
                              '2023-01-03','2023-01-04','first')
        self.assertFalse(label['eligible'])
        self.assertEqual(label['reason'],'origin_contract_absent_at_target')

    def test_exact_origin_and_target_required(self):
        for current,future in [(quote('2023-01-02'),quote('2023-01-04')),
                               (quote('2023-01-03'),quote('2023-01-05'))]:
            with self.assertRaises(ValueError):
                risk.named_label(current,future,'2023-01-03','2023-01-04','second')

    def test_late_calendar_and_missing_table_retained_as_reasons(self):
        current=quote('2023-01-03',publication='2023-01-04T10:30:00')
        late=risk.named_label(current,quote('2023-01-04'),'2023-01-03','2023-01-04','second')
        self.assertEqual(late['reason'],'origin_calendar_assumed_late')
        missing=risk.named_label(None,None,'2023-01-03','2023-01-04','second')
        self.assertEqual(missing['reason'],'missing_exact_origin_table')

    def test_future_value_changes_only_label_not_contract_choice(self):
        current=quote('2023-01-03');future=quote('2023-01-04')
        before=risk.named_label(current,future,'2023-01-03','2023-01-04','second')
        future['quotes']['May-23']=90
        after=risk.named_label(current,future,'2023-01-03','2023-01-04','second')
        self.assertEqual(before['contract'],after['contract'])
        self.assertNotEqual(before['risk'],after['risk'])

    def test_bad_price_is_failure_not_missingness(self):
        future=quote('2023-01-04');future['quotes']['May-23']=0
        with self.assertRaises(ValueError):
            risk.named_label(quote('2023-01-03'),future,'2023-01-03','2023-01-04','second')

    def test_qlike_zero_and_negative_contract(self):
        self.assertEqual(risk.qlike(0,1),0)
        for a,p in [(-1,1),(1,0),(1,-1),(float('nan'),1)]:
            with self.assertRaises(ValueError):risk.qlike(a,p)

    def test_scale_projection_is_conditional_not_a_success_guarantee(self):
        p=risk.scale_projection(2006,.01,.01)
        self.assertGreater(p['conditional_origins_at_observed_effect'],2006)
        self.assertIn('NOT calibrated power',p['status'])
        self.assertIsNone(risk.scale_projection(2006,0,.01)['conditional_origins_at_observed_effect'])


if __name__=='__main__':
    unittest.main()
