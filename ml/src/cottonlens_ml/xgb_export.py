"""Preserve sklearn early-stopping prediction semantics in the native runtime."""


def inference_booster(model):
    booster = model.get_booster()
    best_iteration = booster.attr("best_iteration")
    return booster[:int(best_iteration) + 1] if best_iteration is not None else booster
