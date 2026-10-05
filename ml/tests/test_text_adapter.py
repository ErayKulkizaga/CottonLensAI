"""Synthetic fit/serialization contracts; no Cotton market fitting."""
import json

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.text_adapter import TextPreprocessor


def fixture():
    words = ['texas', 'india', 'china', 'brazil', 'pakistan', 'mali', 'australia', 'drought', 'rain', 'demand', 'supply', 'exports']
    texts = ['Cotton production consumption outlook '+ ' '.join(words[(i+j)%12] for j in (0,1,3,6))
             + f' report {i}' for i in range(12)]
    config = {'column':'text','id_column':'doc_id','max_features':128,'min_df':2,'dimensions':8,
              'minimum_documents':12,'seed':42,'latent_names':[f'svd{i}' for i in range(8)]}
    train = pd.DataFrame({'text':texts*2,'doc_id':[content_id(t) for t in texts]*2,
                          'value':np.arange(24.)})
    return train, ['value',*config['latent_names']], config


def test_document_idf_is_not_daily_repeat_weighted():
    train, names, config = fixture()
    first = TextPreprocessor.fit(train,names,config)
    repeated = pd.concat([train,train.iloc[[0]*20]])
    second = TextPreprocessor.fit(repeated,names,config)
    assert first.vocabulary == second.vocabulary
    np.testing.assert_array_equal(first.idf,second.idf)
    np.testing.assert_array_equal(first.components,second.components)
    assert len(first.training_document_ids) == 12


def test_future_unseen_documents_do_not_fit_or_change_state():
    train, names, config = fixture()
    fitted = TextPreprocessor.fit(train,names,config)
    before = json.dumps(fitted.as_dict(),sort_keys=True)
    future = 'newfuturetoken anotherunseenword 2099'
    test = pd.DataFrame({'text':[future,None],'doc_id':[content_id(future),None],'value':[1.,np.nan]})
    result = fitted.transform(test)
    assert np.isfinite(result).all()
    assert 'newfuturetoken' not in fitted.vocabulary and '2099' not in fitted.vocabulary
    assert json.dumps(fitted.as_dict(),sort_keys=True) == before
    assert result[1,-8:].tolist() == [1.]*8  # Missing text imputed with explicit indicators.


def test_json_state_round_trip_needs_no_refitting():
    train, names, config = fixture()
    fitted = TextPreprocessor.fit(train,names,config)
    body = json.loads(json.dumps(fitted.as_dict()))
    restored = TextPreprocessor.from_dict(body)
    np.testing.assert_array_equal(restored.transform(train),fitted.transform(train))
    assert body['feature_order'] == names+[n+'__missing' for n in names]
    assert fitted.transform(train).shape == (24,18)


def test_manual_serialized_tfidf_matches_sklearn_transform():
    from cottonlens_ml.research.text_adapter import TOKEN
    from sklearn.feature_extraction.text import TfidfVectorizer
    train, names, config = fixture()
    fitted = TextPreprocessor.fit(train,names,config)
    texts = sorted(set(train.text))
    vectorizer = TfidfVectorizer(max_features=128,min_df=2,token_pattern=TOKEN,dtype=np.float64).fit(texts)
    expected = vectorizer.transform(train.text) @ np.asarray(fitted.components).T
    actual = fitted.augment(train)[config['latent_names']].to_numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('change', ['few_docs','identity','components','order'])
def test_invalid_inputs_and_corrupt_state_rejected(change):
    train, names, config = fixture()
    if change == 'few_docs':
        with pytest.raises(ValueError,match='unique eligible'):
            TextPreprocessor.fit(train.iloc[:11],names,config)
    elif change == 'identity':
        train.loc[0,'text'] = 'Changed bytes, same identity'
        with pytest.raises(ValueError,match='identity'):
            TextPreprocessor.fit(train,names,config)
    else:
        body = TextPreprocessor.fit(train,names,config).as_dict()
        if change == 'components':
            body['components'][0][0] = float('nan')
        else:
            body['feature_order'].reverse()
        with pytest.raises(ValueError):
            TextPreprocessor.from_dict(body)


def test_synthetic_model_checkpoint_contains_exact_text_transform(tmp_path,monkeypatch):
    from cottonlens_ml.research import models
    train, names, config = fixture()
    train['date'] = pd.date_range('2017-01-01',periods=len(train))
    train['target_date_5'] = train.date+pd.Timedelta(days=5)
    train['cotton_close'] = 60.
    train['target_return_5'] = np.sin(np.arange(len(train)))*.002
    test = train.iloc[:2].copy()
    test['date'] += pd.Timedelta(days=100)
    spec = {'family':'ridge','horizon':5,'device':'cpu','features':names,'text_adapter':config,
            'target':'scaled_log','seed':42,'params':{'alpha':1},'window':1,'task':'price'}
    monkeypatch.setattr(models,'require_training',lambda *a:None)
    result = models.fit_predict(pd.concat([train,test]),train,None,test,spec,tmp_path)
    state = json.loads((tmp_path/'adapter.json').read_text())
    processor = TextPreprocessor.from_dict(state['processor'])
    linear = json.loads((tmp_path/'model.json').read_text())
    transformed = processor.transform(test) @ np.asarray(linear['coef']) + linear['intercept']
    expected = transformed*state['target']['scale']+state['target']['mean']
    np.testing.assert_allclose(expected,result['predictions'],atol=1e-9,rtol=1e-6)
    assert result['telemetry']['thread_limit'] == 2
