"""
텍스트 분류 서비스
BentoML을 사용하여 텍스트 분류 모델을 서비스화합니다.
"""
import bentoml
import os
import joblib
from typing import List

# 환경변수로 모델 경로 설정 (기본값: 로컬 파일)
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "./vectorizer.pkl")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "./classifier.pkl")
# BentoML 모델 태그 (환경변수로 override 가능)
BENTOML_MODEL_TAG = os.getenv("BENTOML_MODEL_TAG", None)


@bentoml.service(
    image=bentoml.images.Image(python_version="3.11").python_packages("scikit-learn"),
)
class TextClassifier:
    """
    텍스트 분류 서비스 클래스
    
    학습된 벡터화기(TfidfVectorizer)와 분류기(MultinomialNB)를 로드하여
    텍스트 분류 예측을 수행합니다.
    """
    
    def __init__(self) -> None:
        """
        모델 및 벡터화기 초기화 및 로드
        
        로드 우선순위:
        1. BentoML 모델 저장소 (BENTOML_MODEL_TAG 환경변수 설정 시)
        2. 로컬 파일 (joblib으로 저장된 .pkl 파일)
        
        Raises:
            FileNotFoundError: 모델 파일을 찾을 수 없는 경우
            Exception: 모델 로드 중 오류 발생 시
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        
        # BentoML 모델 저장소에서 로드 시도
        if BENTOML_MODEL_TAG:
            try:
                model_ref = bentoml.models.get(BENTOML_MODEL_TAG)
                self.vectorizer = joblib.load(model_ref.path_of("vectorizer.pkl"))
                self.classifier = joblib.load(model_ref.path_of("classifier.pkl"))
                print(f"✅ BentoML 모델 저장소에서 로드 성공: {BENTOML_MODEL_TAG}")
                return
            except Exception as e:
                print(f"⚠️ BentoML 모델 저장소 로드 실패: {e}, 파일 로드 시도...")
        
        # 로컬 파일에서 로드 (fallback)
        try:
            if os.path.exists(VECTORIZER_PATH):
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                print(f"✅ 벡터화기 로드 성공: {VECTORIZER_PATH}")
            else:
                print(f"⚠️ 벡터화기 파일 없음: {VECTORIZER_PATH}, 새로 생성합니다.")
                self.vectorizer = TfidfVectorizer()
            
            if os.path.exists(CLASSIFIER_PATH):
                self.classifier = joblib.load(CLASSIFIER_PATH)
                print(f"✅ 분류기 로드 성공: {CLASSIFIER_PATH}")
            else:
                print(f"⚠️ 분류기 파일 없음: {CLASSIFIER_PATH}, 새로 생성합니다.")
                self.classifier = MultinomialNB()
                # 테스트용: 예시 데이터로 학습 (실전에서는 제거 필요)
                self._train_with_sample_data()
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _train_with_sample_data(self) -> None:
        """
        예시 데이터로 모델 학습 (테스트용)
        
        주의: 실전에서는 이 메서드를 제거하고 
        독립적인 학습 스크립트에서 모델을 학습/저장해야 합니다.
        """
        sample_texts = [
            "This is a positive example.",
            "This is a negative example.",
            "Great product, would buy again!",
            "Worst experience ever.",
        ]
        sample_labels = [1, 0, 1, 0]
        vectors = self.vectorizer.fit_transform(sample_texts)
        self.classifier.fit(vectors, sample_labels)
        print("⚠️ 예시 데이터로 학습 완료 (테스트용)")

    @bentoml.api(batchable=True)
    def predict(self, texts: List[str]) -> List[int]:
        """
        텍스트 분류 예측 수행
        
        Args:
            texts: 분류할 텍스트 리스트
            
        Returns:
            분류 결과 리스트 (0: negative, 1: positive)
            
        Raises:
            ValueError: 입력 텍스트가 비어있는 경우
        """
        if not texts:
            raise ValueError("입력 텍스트 리스트가 비어있습니다.")
        
        # 텍스트를 벡터로 변환
        vectors = self.vectorizer.transform(texts)
        # 분류 예측 수행
        preds = self.classifier.predict(vectors)
        return preds.tolist()