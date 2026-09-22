from cottonlens_ml.xgb_export import inference_booster


def test_native_export_keeps_only_best_iteration():
    class Booster:
        def attr(self, name):
            assert name == "best_iteration"
            return "3"

        def __getitem__(self, item):
            assert item == slice(None, 4)
            return "selected trees"

    class Model:
        def get_booster(self):
            return Booster()

    assert inference_booster(Model()) == "selected trees"
