"""Train-only, serializable TF-IDF/SVD adapter for the bounded WASDE pilot."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.protocol import Preprocessor

TOKEN = r'(?u)\b[a-zA-Z][a-zA-Z]+\b'


@dataclass
class TextPreprocessor:
    config: dict
    vocabulary: dict
    idf: list
    components: list
    processor: Preprocessor
    training_document_ids: list

    @staticmethod
    def documents(rows, config):
        values = []
        for text, identity in zip(rows[config['column']], rows[config['id_column']], strict=True):
            if pd.isna(text):
                if not pd.isna(identity):
                    raise ValueError('Missing narrative has a document identity')
                values.append(None)
            elif not isinstance(text, str) or not text.strip() or content_id(text) != identity:
                raise ValueError('Narrative identity/content mismatch')
            else:
                values.append(text)
        return values

    @classmethod
    def fit(cls, train, names, config):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        if (config['max_features'] != 128 or config['min_df'] != 2
                or config['dimensions'] != 8 or config['minimum_documents'] != 12
                or config['seed'] != 42 or len(config['latent_names']) != 8
                or list(names[-8:]) != config['latent_names']):
            raise ValueError('Only the fixed registered text transform is permitted')
        texts = cls.documents(train, config)
        unique = {content_id(text): text for text in texts if text is not None}
        ids = sorted(unique)
        if len(ids) < 12:
            raise ValueError('At least 12 unique eligible training documents required')
        vectorizer = TfidfVectorizer(max_features=128, min_df=2, token_pattern=TOKEN,
                                   lowercase=True, ngram_range=(1, 1), norm='l2', dtype=np.float64)
        matrix = vectorizer.fit_transform([unique[key] for key in ids])
        if matrix.shape[1] < 8:
            raise ValueError('Insufficient training vocabulary for fixed SVD8')
        svd = TruncatedSVD(n_components=8, n_iter=7, random_state=42).fit(matrix)
        vocabulary = {word: int(index) for word, index in vectorizer.vocabulary_.items()}
        result = cls(dict(config), vocabulary, vectorizer.idf_.tolist(),
                     svd.components_.tolist(), None, ids)
        augmented = result.augment(train)
        result.processor = Preprocessor.fit(augmented, names)
        return result

    def augment(self, rows):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize
        texts = self.documents(rows, self.config)
        unique = sorted({text for text in texts if text is not None})
        vectors = {}
        if unique:
            counts = CountVectorizer(vocabulary=self.vocabulary, token_pattern=TOKEN,
                                     lowercase=True, ngram_range=(1, 1)).transform(unique)
            tfidf = normalize(counts.astype(np.float64).multiply(np.asarray(self.idf)), norm='l2')
            latent = np.asarray(tfidf @ np.asarray(self.components).T)
            vectors = dict(zip(unique, latent, strict=True))
        features = np.asarray([vectors[text] if text is not None else np.full(8, np.nan)
                               for text in texts]).reshape(len(rows), 8)
        result = rows.copy()
        result[self.config['latent_names']] = features
        return result

    def transform(self, rows):
        return self.processor.transform(self.augment(rows))

    def as_dict(self):
        return {'kind': 'wasde-tfidf-svd-v1', 'config': self.config,
                'vocabulary': self.vocabulary, 'idf': self.idf, 'components': self.components,
                'numeric': self.processor.as_dict(), 'training_document_ids': self.training_document_ids,
                'feature_order': [*self.processor.names, *[name+'__missing' for name in self.processor.names]]}

    @classmethod
    def from_dict(cls, body):
        if body['kind'] != 'wasde-tfidf-svd-v1':
            raise ValueError('Unsupported text adapter schema')
        vocabulary, idf, components = body['vocabulary'], body['idf'], body['components']
        numeric = body['numeric']
        if (sorted(vocabulary.values()) != list(range(len(vocabulary)))
                or np.asarray(idf).shape != (len(vocabulary),)
                or np.asarray(components).shape != (8, len(vocabulary))
                or not np.isfinite(idf).all() or not np.isfinite(components).all()
                or np.any(np.asarray(idf) <= 0)
                or any(len(numeric[key]) != len(numeric['names']) for key in ('median','mean','scale'))
                or not np.isfinite([numeric['median'],numeric['mean'],numeric['scale']]).all()
                or np.any(np.asarray(numeric['scale']) <= 0)
                or numeric['names'][-8:] != body['config']['latent_names']):
            raise ValueError('Corrupt text adapter state')
        result = cls(body['config'], vocabulary, idf, components, Preprocessor(**numeric),
                     body['training_document_ids'])
        if result.as_dict()['feature_order'] != body['feature_order']:
            raise ValueError('Text feature order changed')
        return result
