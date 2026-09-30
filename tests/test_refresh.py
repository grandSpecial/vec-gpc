import unittest
from unittest.mock import Mock
from types import SimpleNamespace as NS
from sqlalchemy.dialects import postgresql
from refresh_embeddings import pending_embeddings, save_batch
from embedding_text import build_gpc_embedding_text, embedding_revision
from taxonomy import BRICK_PATHS
from test_classifier import brick


class RefreshTests(unittest.TestCase):
    def test_only_changed_or_missing_vectors_are_refreshed(self):
        item=brick()
        text=build_gpc_embedding_text(item.title,' > '.join(BRICK_PATHS[item.code]),item.definition)
        revision=embedding_revision(text)
        db=Mock();db.execute.return_value.all.return_value=[(item,revision,item.id)]
        self.assertEqual(pending_embeddings(db),[])
        db.execute.return_value.all.return_value=[(item,revision,None)]
        self.assertEqual(len(pending_embeddings(db)),1)
        db.execute.return_value.all.return_value=[(item,'gpc-embedding-v1',item.id)]
        self.assertEqual(len(pending_embeddings(db)),1)

    def test_vector_and_marker_share_callers_transaction(self):
        db=Mock()
        save_batch(db,[(1,'product','revision')],[[0.0]*1536])
        self.assertEqual(db.execute.call_count,2)
        db.commit.assert_not_called()
        sql=[str(c.args[0].compile(dialect=postgresql.dialect())) for c in db.execute.call_args_list]
        self.assertIn('INSERT INTO items',sql[0])
        self.assertIn('INSERT INTO embedding_refresh_state',sql[1])

    def test_incomplete_embedding_response_fails_before_writes(self):
        db=Mock()
        with self.assertRaises(ValueError):save_batch(db,[(1,'product','revision')],[])
        db.execute.assert_not_called()
        with self.assertRaises(ValueError):save_batch(db,[(1,'product','revision')],[[.1]])
        db.execute.assert_not_called()


if __name__=='__main__':unittest.main()
