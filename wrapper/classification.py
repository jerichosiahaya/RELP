from predict.predict import contextual_prediction
from config.log import log

class Classification:
    def __init__(self, embedding_store):
        self.embedding_store = embedding_store
        self.documents, self.vectors = self.embedding_store.load()

    def predict(self, content: str) -> int:
        # search vector similarity
        similar_articles = self.embedding_store.query(self.documents, self.vectors, content)

        # predict the context
        prediction = contextual_prediction(similar_articles, content)

        return int(prediction)