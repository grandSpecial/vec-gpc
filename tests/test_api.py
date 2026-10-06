import os
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
os.environ.setdefault('OPENAI_API_KEY', 'test')
os.environ.setdefault('API_AUTH_TOKEN', 'test')

import ast
import asyncio
import logging
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import httpx
from fastapi.testclient import TestClient
import main
from test_classifier import candidate


def result():
    return {
        'id':10005897,'code':10005897,'title':'Bananas','full_title':'Food/Beverage Bananas',
        'level_2_category':'Produce','level_3_category':'Bananas','category':'Produce',
        'subcategory':'Bananas','display_label':'Bananas','description':'A banana.',
        'input_text':'Banana','normalized_text':'banana','normalization':{'version':'test','expansions':[]},
        'definition':'','active':True,'confidence':.41,'status':'classified','needs_review':True,
        'display_mapping_version':'display-mapping-v1','display_mapping_source':'rules',
        'reranker_version':'test','latency_ms':1,'description_fallback':False,
        'log_candidates':[candidate()], 'selection_source':'model','selection_fallback':False,
    }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.db=Mock()
        main.app.dependency_overrides[main.get_db]=lambda:self.db
        self.classifier=patch.object(main.GPCClassifier,'classify',side_effect=lambda *a,**k:result())
        self.classify=self.classifier.start()
        self.logging=patch.object(main,'log_classification_event')
        self.log=self.logging.start()
        self.client=TestClient(main.app)
        self.headers={'Authorization':f'Bearer {main.API_AUTH_TOKEN}'}

    def tearDown(self):
        self.client.close();self.classifier.stop();self.logging.stop()
        main.app.dependency_overrides.clear()

    def test_legacy_request_and_response(self):
        response=self.client.post('/search',params={'text':'Banana'},headers=self.headers)
        self.assertEqual(response.status_code,200)
        data=response.json()
        self.assertEqual(data['level_2_category'],'Produce')
        self.assertEqual(data['category'],data['level_2_category'])
        self.assertNotIn('log_candidates',data)
        self.assertEqual(data['code'],10005897)
        self.db.close.assert_called_once()

    def test_optional_json_form_and_query_precedence(self):
        self.assertEqual(self.client.post('/search',json={'text':'Banana'},headers=self.headers).status_code,200)
        self.client.post('/search',params={'text':'Banana'},json={'text':'Apple'},headers=self.headers)
        self.assertEqual(self.classify.call_args.args[0],'Banana')

    def test_deposit_response_does_not_require_gpc_or_product_log(self):
        from receipt_charges import receipt_charge_result
        from normalization import normalize_receipt_text
        data = receipt_charge_result(normalize_receipt_text('Bottle deposit'), True)
        data.update(reranker_version='test', latency_ms=0)
        self.classify.side_effect = lambda *a, **k: data.copy()
        response = self.client.post('/search', params={'text':'Bottle deposit'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['level_2_category'], 'Deposits & Fees')
        self.assertEqual(response.json()['status'], 'non_product')
        self.assertIsNone(response.json()['code'])
        self.assertNotIn('log_candidates', response.json())
        self.log.assert_not_called()
        self.db.close.assert_called_once()

    def test_input_validation_and_auth(self):
        for params in [{},{'text':''},{'text':'   '},{'text':'x'*2049}]:
            self.assertEqual(self.client.post('/search',params=params,headers=self.headers).status_code,422)
        self.assertIn(self.client.post('/search',params={'text':'Banana'}).status_code,[401,403])
        self.assertEqual(self.client.post('/search',params={'text':'Banana'},headers={'Authorization':'Bearer incorrect'}).status_code,401)

    def test_errors_do_not_leak_details(self):
        self.classify.side_effect=RuntimeError('private credentials')
        response=self.client.post('/search',params={'text':'Banana'},headers=self.headers)
        self.assertEqual(response.status_code,500)
        self.assertNotIn('private credentials',response.text)

    def test_consumer_function_against_asgi(self):
        # Extract only the real consumer helper: importing its app has unrelated side effects.
        source=Path(__file__).with_name('fixtures').joinpath('gouge_busters_category_client.py').read_text()
        module=ast.parse(source)
        function=next(n for n in module.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='get_category_from_api')
        namespace={'os':os,'httpx':httpx,'logger':logging.getLogger('consumer')}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<consumer>','exec'),namespace)
        original=httpx.AsyncClient
        def client(**kwargs):
            return original(transport=httpx.ASGITransport(app=main.app),**kwargs)
        with patch.dict(os.environ,{'VEC_GPC_API_KEY':main.API_AUTH_TOKEN}),patch.object(httpx,'AsyncClient',side_effect=client):
            value=asyncio.run(namespace['get_category_from_api']('Banana'))
        self.assertEqual(value,'Produce')



class ProviderFailureTests(unittest.TestCase):
    def test_embedding_timeout_is_generic_504(self):
        import openai
        main.create_vector.cache_clear()
        error=openai.APITimeoutError(request=httpx.Request('POST','https://api.openai.com/v1/embeddings'))
        with patch.object(main.client,'with_options',return_value=main.client), patch.object(main.client.embeddings,'create',side_effect=error):
            with self.assertRaises(main.HTTPException) as caught:
                main.create_vector('timeout test')
        self.assertEqual(caught.exception.status_code,504)
        self.assertNotIn('api.openai.com',caught.exception.detail)

    def test_log_write_failure_does_not_escape(self):
        db=Mock();db.commit.side_effect=RuntimeError('database unavailable')
        row=candidate()
        with patch.object(main,'SessionLocal',return_value=db):
            main.log_classification_event('Banana','Banana',row.gpc_item,'Produce','Bananas',.5,[row],1)
        db.rollback.assert_called_once()
        db.close.assert_called_once()

if __name__ == '__main__': unittest.main()
