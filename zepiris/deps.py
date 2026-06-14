from typing import Annotated

from fastapi import Depends, Request

from zepiris.config import Settings, get_settings
from zepiris.services.embedding import FaceEmbeddingProvider
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.learning import AdaptiveThresholdLearner
from zepiris.services.s3_fetcher import S3ImageFetcher


def settings_dep() -> Settings:
    return get_settings()


def iqa_dep(request: Request) -> MLInferenceIQAService:
    return request.app.state.iqa


def embedding_dep(request: Request) -> FaceEmbeddingProvider:
    return request.app.state.embedding


def s3_fetcher_dep(request: Request) -> S3ImageFetcher:
    return request.app.state.s3_fetcher


def learner_dep(request: Request) -> AdaptiveThresholdLearner:
    return request.app.state.learner


SettingsDep = Annotated[Settings, Depends(settings_dep)]
IQADep = Annotated[MLInferenceIQAService, Depends(iqa_dep)]
EmbeddingDep = Annotated[FaceEmbeddingProvider, Depends(embedding_dep)]
S3FetcherDep = Annotated[S3ImageFetcher, Depends(s3_fetcher_dep)]
LearnerDep = Annotated[AdaptiveThresholdLearner, Depends(learner_dep)]
