import json
import requests
from requests.exceptions import Timeout, ConnectionError
import numpy as np
import re
import time
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Ollama API 엔드포인트
OLLAMA_API = "http://localhost:11434/api/generate"

# 임베딩 생성을 위한 대체 함수 정의
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 임베딩 모델 로드 - SentenceTransformer 대신 Hugging Face 모델 직접 사용
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embedding_model_base = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 임베딩 함수 정의
def get_embeddings(texts):
    # 배치 처리
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # 모델 출력 가져오기
    with torch.no_grad():
        model_output = embedding_model_base(**encoded_input)
    
    # 평균 풀링으로 임베딩 생성
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # 정규화
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.numpy()

# 타임아웃 설정 (초)
REQUEST_TIMEOUT = 30

def load_data(file_path):
    """평가 데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def query_ollama(sentence, model_name="cusmo7"): #cusmo4
    """Ollama API를 사용하여 모델 평가 획득 (타임아웃 적용)"""
    # 시스템 메시지와 유저 메시지를 학습 형식과 동일하게 구성
    system_msg = "당신은 문법 검사기입니다. 입력된 문장의 문법이 맞는지 분석하세요."
    user_prompt = f"문장: {sentence}\n문법적으로 정확한가요?"
    
    # 모델파일 템플릿 형식에 맞게 구성
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # 타임아웃 적용한 API 요청
        response = requests.post(OLLAMA_API, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()["response"], False  # 두 번째 값은 타임아웃 여부
        else:
            print(f"API 오류: {response.status_code}")
            return None, False
            
    except Timeout:
        print(f"\r타임아웃 발생 ({REQUEST_TIMEOUT}초 초과)", end="")
        return None, True  # 타임아웃 발생 표시
    except ConnectionError:
        print(f"\r연결 오류 발생", end="")
        return None, False
    except Exception as e:
        print(f"\r요청 중 오류 발생: {e}", end="")
        return None, False

def parse_model_response(response):
    """모델 응답 파싱 - 학습 형식에 맞게 강화"""
    answer = ""
    rational = ""
    
    # <think> 태그 내용은 파싱에서 제외 (사고 과정)
    think_pattern = r'<think>.*?</think>'
    response_without_think = re.sub(think_pattern, '', response, flags=re.DOTALL)
    
    # 정형화된 답변 형식 파싱
    answer_match = re.search(r'답\s*:\s*(맞음|아님)', response_without_think)
    rational_match = re.search(r'이유\s*:\s*(.*?)(?:\n|$)', response_without_think, re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    
    if rational_match:
        rational = rational_match.group(1).strip()
    
    # 정규식 실패 시 추가 검색 방법
    if not answer:
        if "맞음" in response_without_think:
            answer = "맞음"
        elif "아님" in response_without_think:
            answer = "아님"
    
    # 명확하게 맞음/아님이 식별되었는지 여부도 반환
    has_answer = answer in ["맞음", "아님"]
    
    return {
        "answer": answer,
        "rational": rational,
        "has_answer": has_answer
    }

def calculate_accuracy(eval_data, model_outputs):
    """방식 1: 맞음/아님 정확도 계산"""
    correct = 0
    total = len(eval_data)
    
    for i, item in enumerate(eval_data):
        if i < len(model_outputs) and item["answer"] == model_outputs[i]["answer"]:
            correct += 1
    
    return correct, total, correct / total * 100 if total > 0 else 0

def calculate_answer_relevancy(question, rational):
    """질문과 답변 이유 사이의 관련성 계산 (임베딩 기반 코사인 유사도)"""
    if not question or not rational:
        return 0
    
    # 질문과 답변 이유를 임베딩 벡터로 변환
    try:
        # 수정된 임베딩 함수 사용
        question_embedding = get_embeddings([question])[0]
        rational_embedding = get_embeddings([rational])[0]
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([question_embedding], [rational_embedding])[0][0]
        return similarity
    except Exception as e:
        print(f"임베딩 계산 오류: {e}")
        return 0

def calculate_answer_similarity(ref_rational, model_rational):
    """답변 유사도 계산 (임베딩 기반 코사인 유사도)"""
    if not ref_rational or not model_rational:
        return 0
    
    # <think> 태그와 그 내용 제거
    think_pattern = r'<think>.*?</think>'
    model_rational_clean = re.sub(think_pattern, '', model_rational, flags=re.DOTALL)
    
    # 문장 임베딩 계산
    try:
        ref_embedding = get_embeddings([ref_rational])[0]
        model_embedding = get_embeddings([model_rational_clean])[0]
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([ref_embedding], [model_embedding])[0][0]
        return similarity
    except Exception as e:
        print(f"임베딩 계산 오류: {e}")
        return 0

def calculate_metrics(eval_data, model_outputs):
    """방식 2: 텍스트 유사도 메트릭 계산"""
    answer_relevancy_scores = []
    answer_similarity_scores = []
    rouge_scores = []
    meteor_scores = []
    
    # ROUGE 스코어러 초기화
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for i, item in enumerate(eval_data):
        if i >= len(model_outputs):
            break
            
        # ground truth와 모델 출력
        ref_answer = item["answer"]
        ref_rational = item["rational"]
        question = item["sentence"]
        model_answer = model_outputs[i]["answer"]
        model_rational = model_outputs[i]["rational"]
        
        # <think> 태그 내용 제거 (추가 보장)
        think_pattern = r'<think>.*?</think>'
        if model_rational:
            model_rational = re.sub(think_pattern, '', model_rational, flags=re.DOTALL)
        
        # 1. Answer Relevancy (질문과 답변 이유의 관련성)
        relevancy = calculate_answer_relevancy(question, model_rational)
        answer_relevancy_scores.append(relevancy)
        
        # 2. Answer Similarity (임베딩 기반 - 참조 이유와 모델 이유 비교)
        if ref_rational and model_rational:
            similarity = calculate_answer_similarity(ref_rational, model_rational)
            answer_similarity_scores.append(similarity)
        
        # 3. ROUGE-L 점수
        if ref_rational and model_rational:
            rouge = scorer.score(ref_rational, model_rational)
            rouge_scores.append(rouge['rougeL'].fmeasure)
        
        # 4. METEOR 점수
        if ref_rational and model_rational:
            ref_tokens = ref_rational.lower().split()
            model_tokens = model_rational.lower().split()
            meteor = meteor_score([ref_tokens], model_tokens)
            meteor_scores.append(meteor)
    
    # 결과 계산
    metrics = {
        "answer_relevancy": np.mean(answer_relevancy_scores) if answer_relevancy_scores else 0,
        "answer_similarity": np.mean(answer_similarity_scores) if answer_similarity_scores else 0,
        "rouge_l": np.mean(rouge_scores) if rouge_scores else 0,
        "meteor": np.mean(meteor_scores) if meteor_scores else 0
    }
    
    return metrics

def main():
    # 데이터 로드
    try:
        eval_data = load_data("eval_data.json")
        print(f"총 {len(eval_data)}개의 평가 데이터를 로드했습니다.")
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return
    
    # 모델 평가 수행
    model_outputs = []
    for i, item in enumerate(eval_data):
        print(f"\r문장 평가 중... {i+1}/{len(eval_data)}", end="")
        
        # 최대 5번 시도
        max_attempts = 5
        attempts = 0
        parsed = None
        
        while attempts < max_attempts:
            # Ollama API 호출
            response, is_timeout = query_ollama(item["sentence"])
            
            # 응답이 있고 타임아웃이 아닌 경우
            if response:
                parsed = parse_model_response(response)
                
                # 맞음/아님 응답이 포함되어 있으면 루프 종료
                if parsed["has_answer"]:
                    break
                else:
                    attempts += 1
                    print(f"\r문장 평가 중... {i+1}/{len(eval_data)} (재시도 {attempts}/{max_attempts})", end="")
            else:
                # 타임아웃이나 응답 없음 - 다음 시도
                attempts += 1
                if is_timeout:
                    print(f"\r문장 평가 중... {i+1}/{len(eval_data)} (타임아웃 발생, 재시도 {attempts}/{max_attempts})", end="")
                    # 타임아웃 발생 시 잠시 대기 후 재시도
                    time.sleep(1)
                else:
                    print(f"\r문장 평가 중... {i+1}/{len(eval_data)} (응답 없음, 재시도 {attempts}/{max_attempts})", end="")
                
        # 최대 시도 후에도 맞음/아님 응답이 없으면 "아님"으로 처리
        if not parsed or not parsed["has_answer"]:
            parsed = {
                "answer": "아님",  # 기본값으로 "아님" 설정
                "rational": "모델이 명확한 답변을 제공하지 않음",
                "has_answer": False
            }
            print(f"\r문장 평가 중... {i+1}/{len(eval_data)} (최대 시도 후 실패, '아님'으로 처리)", end="")
        
        model_outputs.append({
            "sentence": item["sentence"],
            "answer": parsed["answer"],
            "rational": parsed["rational"],
            "full_response": response if response else "응답 없음"
        })
    
    print("\n평가 완료!")
    
    # 방식 1: 정확도 평가
    correct, total, accuracy = calculate_accuracy(eval_data, model_outputs)
    print(f"\n=== 방식 1: 정확도 평가 ===")
    print(f"정확도: {accuracy:.2f}% ({correct}/{total})")
    
    # 방식 2: 텍스트 유사도 메트릭 계산
    metrics = calculate_metrics(eval_data, model_outputs)
    print(f"\n=== 방식 2: 텍스트 유사도 평가 ===")
    print(f"Answer Relevancy: {metrics['answer_relevancy']:.4f}")
    print(f"Answer Similarity: {metrics['answer_similarity']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    
    # 저장
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "metrics": metrics,
            "model_outputs": model_outputs
        }, f, ensure_ascii=False, indent=2)
    
    print("\n평가 결과가 evaluation_results.json에 저장되었습니다.")

if __name__ == "__main__":
    # 필요 라이브러리 체크
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except ImportError:
        print("필요한 NLTK 데이터를 설치해주세요: pip install nltk")
    
    main()
