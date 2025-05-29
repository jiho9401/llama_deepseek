import json
import os
import time
import re
import threading
import sys
import torch
import requests
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_generator.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 상수 정의
MAIN_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 티처 모델
PARSING_MODEL = "deepseek-r1:8b"  # Ollama 모델명
SERVER2_API = "http://192.168.0.1:5000/receive_data"  # 서버 2 API 엔드포인트
TOP_K = 100  # 각 위치에서 상위 k개 토큰만 전송

# 데이터 큐 및 전송 상태 관리
data_queue = []
QUEUE_SIZE_THRESHOLD = 10  # 이 개수만큼 쌓이면 서버 2로 전송
last_transmission_time = 0
TRANSMISSION_INTERVAL = 300  # 5분마다 데이터 전송 (초 단위)

# Ollama API 설정
OLLAMA_API_BASE = "http://localhost:11434/api"

def create_main_prompt(sentence):
    """메인 평가를 위한 프롬프트 생성"""
    prompt = f"""다음의 평가해야할 문장을 보고서 해당 형식에 맞도록 한국어로 답변하시오 
    rational은 해당 답변의 이유를 한국어로 작성하고 예시를 들기 위해서 영어를 기입해도 된다.
"sentence": "문장",
"answer": "답변(맞음, 아님)",
"rational": "해당 답변의 이유"

평가해야할 문장은 다음과 같습니다.
평가해야할 문장: {sentence}"""
    return prompt

def wait_spinner(stop_event, model_name):
    """대기 중인 상태와 경과 시간을 표시하는 스피너"""
    start_time = time.time()
    spinner = ['|', '/', '-', '\\']
    idx = 0
    
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r{model_name} 모델 응답 대기 중 {spinner[idx]} ({elapsed:.1f}초 경과)...")
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner)
        time.sleep(0.1)
    
    # 스피너 라인 지우기
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def get_model_tokenizer(self, model_name, device="auto"):
        """모델과 토크나이저를 로드하거나 캐시된 것을 반환"""
        if model_name not in self.models:
            # 스피너 시작
            stop_event = threading.Event()
            spinner_thread = threading.Thread(target=wait_spinner, args=(stop_event, f"{model_name} (로딩 중)"))
            spinner_thread.daemon = True
            spinner_thread.start()
            
            try:
                logger.info(f"{model_name} 모델 및 토크나이저 로딩 시작")
                start_time = time.time()
                
                # 토크나이저 로드
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                logger.info(f"{model_name} 토크나이저 로드 완료")
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 32B 모델은 특별 처리
                if "32B" in model_name and "deepseek" in model_name.lower():
                    logger.info(f"대용량 모델({model_name}) 로드 중 - 특별 처리 적용")
                    
                    # 4비트 양자화 설정
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    
                    # 모델 로드 시도
                    try:
                        logger.info("방법 1: 전체 모델을 GPU 0에 로드 시도...")
                        self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=quantization_config,
                            device_map={"": 0},
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            use_flash_attention_2=False
                        )
                        logger.info("방법 1: 32B 모델 로드 성공! (GPU 0에 전체 배치)")
                    except Exception as e1:
                        logger.warning(f"방법 1 실패: {e1}")
                        
                        # 대체 모델 시도
                        alternate_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                        logger.info(f"큰 모델(32B) 로드 실패. 대체 모델 시도: {alternate_model}")
                        try:
                            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                                alternate_model,
                                quantization_config=quantization_config,
                                device_map="auto",
                                trust_remote_code=True,
                                torch_dtype=torch.float16,
                                use_flash_attention_2=False
                            )
                            logger.info(f"대체 모델({alternate_model}) 로드 성공!")
                        except Exception as e2:
                            logger.error(f"대체 모델 로드 실패: {e2}")
                            raise
                else:
                    # 다른 모델 (8B 등) - 표준 방식으로 로드
                    logger.info(f"일반 모델({model_name}) 로드 중...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        use_flash_attention_2=False
                    )
                
                elapsed = time.time() - start_time
                logger.info(f"{model_name} 모델 로드 완료! (소요 시간: {elapsed:.2f}초)")
                
                # 스피너 종료
                stop_event.set()
                spinner_thread.join(0.5)
                
            except Exception as e:
                # 스피너 종료
                stop_event.set()
                spinner_thread.join(0.5)
                
                logger.error(f"{model_name} 모델 로드 중 오류: {e}")
                raise
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def get_target_tokenizer(self):
        """타겟 모델의 토크나이저만 로드 (검증용)"""
        if "target_tokenizer" not in self.tokenizers:
            try:
                logger.info(f"\n타겟 모델 토크나이저 로딩 중...")
                self.tokenizers["target_tokenizer"] = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    trust_remote_code=True
                )
                logger.info("타겟 모델 토크나이저 로딩 완료!")
            except Exception as e:
                logger.error(f"타겟 모델 토크나이저 로딩 오류: {e}")
                # 에러가 발생해도 계속 진행 (검증용이므로)
                pass
        
        return self.tokenizers.get("target_tokenizer", None)

    def generate_response(self, model_name, prompt, max_tokens=512):
        """모델 응답 생성 및 top-k 확률 추출"""
        model, tokenizer = self.get_model_tokenizer(model_name)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 응답 생성
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True  # 확률 점수 반환
            )
        
        # 생성된 텍스트 디코딩
        generated_ids = outputs.sequences
        response_text = tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # top-k 토큰 및 확률 추출
        # 생성 과정의 각 단계에서의 확률 분포 가져오기
        probs_at_each_step = []
        for step_idx, scores in enumerate(outputs.scores):
            step_probs = torch.softmax(scores, dim=-1)
            top_k_probs, top_k_indices = torch.topk(step_probs, k=TOP_K, dim=-1)
            
            step_tokens = []
            for batch_idx in range(top_k_indices.shape[0]):
                tokens = []
                for i in range(TOP_K):
                    token_id = top_k_indices[batch_idx, i].item()
                    token_prob = top_k_probs[batch_idx, i].item()
                    token_text = tokenizer.decode([token_id])
                    tokens.append({
                        "token_id": token_id,
                        "token_text": token_text,
                        "probability": token_prob
                    })
                step_tokens.append(tokens)
            probs_at_each_step.append(step_tokens)
        
        return {
            "text": response_text,
            "token_probs": probs_at_each_step
        }

def get_main_model_response(prompt, timeout=30):
    """Transformers 모델을 사용하여 응답 생성 (logits 포함)"""
    # 모델과 토크나이저 가져오기
    model, tokenizer = model_manager.get_model_tokenizer(MAIN_MODEL)
    
    # 스피너 스레드 설정
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=wait_spinner, args=(stop_event, MAIN_MODEL))
    spinner_thread.daemon = True
    spinner_thread.start()
    
    try:
        logger.info("메인 모델 추론 시작...")
        start_time = time.time()
        
        # 입력 인코딩
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 생성 설정
        gen_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0.2,  # 낮은 temperature로 일관성 있는 응답 생성
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id  # EOS 토큰을 패딩 토큰으로 사용
        }
        
        # 타임아웃 처리를 위한 시간 체크
        current_time = time.time()
        if current_time - start_time > timeout:
            stop_event.set()
            spinner_thread.join(0.5)
            logger.warning(f"메인 모델 응답 타임아웃! 제한 시간({timeout}초)이 초과되었습니다.")
            return None
        
        # 모델 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_len = len(prompt)
        response_text = generated_text[prompt_len:]
        
        # 스피너 중지
        stop_event.set()
        spinner_thread.join(0.5)
        
        # 소요 시간 측정
        elapsed = time.time() - start_time
        logger.info(f"메인 모델 응답 완료! (소요 시간: {elapsed:.2f}초)")
        
        return {"text": response_text}
    
    except KeyboardInterrupt:
        # 스피너 중지
        stop_event.set()
        spinner_thread.join(0.5)
        logger.warning("사용자에 의해 중단됨")
        raise
    
    except Exception as e:
        # 스피너 중지
        stop_event.set()
        spinner_thread.join(0.5)
        logger.error(f"메인 모델 응답 생성 중 오류: {e}")
        return None

def get_parsing_model_response(raw_response_data, timeout=30):
    """Ollama API를 사용하여 파싱 응답 생성"""
    # 원본 응답 텍스트 추출
    raw_response = raw_response_data["text"] if isinstance(raw_response_data, dict) else raw_response_data
    
    # 파싱용 프롬프트 생성
    template = """다음 텍스트에서 "answer"와 "rational" 필드의 값을 추출해주세요.
텍스트는 JSON 형식이 아닐 수 있으므로, 정확한 값만 추출하여 다음 형식으로 응답해주세요:

"answer": "추출한 answer 값"
"rational": "추출한 rational 값"

추출할 텍스트:
{raw_response}"""
    
    prompt = template.format(raw_response=raw_response)
    
    # 스피너 스레드 설정
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=wait_spinner, args=(stop_event, PARSING_MODEL))
    spinner_thread.daemon = True
    spinner_thread.start()
    
    try:
        logger.info("파싱 모델(Ollama) 호출 시작...")
        start_time = time.time()
        
        # Ollama API 호출
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": PARSING_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # 결정적 출력
                    "num_predict": 512,  # 최대 토큰 수
                }
            },
            timeout=timeout  # 요청 타임아웃
        )
        
        # 타임아웃 체크
        elapsed = time.time() - start_time
        if elapsed > timeout:
            stop_event.set()
            spinner_thread.join(0.5)
            logger.warning(f"파싱 모델 응답 타임아웃! 제한 시간({timeout}초)이 초과되었습니다.")
            return None
        
        # 응답 확인
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            
            # 스피너 중지
            stop_event.set()
            spinner_thread.join(0.5)
            
            logger.info(f"파싱 모델 응답 완료! (소요 시간: {elapsed:.2f}초)")
            return generated_text
        else:
            stop_event.set()
            spinner_thread.join(0.5)
            logger.error(f"Ollama API 오류: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        # 타임아웃 예외 발생
        stop_event.set()
        spinner_thread.join(0.5)
        logger.warning(f"파싱 모델 응답 타임아웃! 제한 시간({timeout}초)이 초과되었습니다.")
        return None
    except Exception as e:
        # 기타 예외 발생
        stop_event.set()
        spinner_thread.join(0.5)
        logger.error(f"Ollama API 요청 중 오류 발생: {e}")
        return None

def remove_think_tags(text):
    """<think> 태그와 그 내용 제거"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def parse_response(response):
    """모델 응답에서 answer와 rational 추출"""
    # 응답이 딕셔너리인 경우 텍스트 추출
    if isinstance(response, dict):
        response_text = response["text"]
    else:
        response_text = response
    
    # <think> 태그 제거
    clean_response = remove_think_tags(response_text)
    
    try:
        lines = clean_response.split("\n")
        answer = None
        rational = None
        
        for line in lines:
            if '"answer":' in line:
                answer = line.split('"answer":')[1].strip().strip('",')
            elif '"rational":' in line:
                rational = line.split('"rational":')[1].strip().strip('",')
        
        # 파싱 결과 확인
        if not answer or not rational:
            logger.error("\n===== 파싱 실패 =====")
            logger.error("원인: answer 또는 rational 필드를 찾을 수 없습니다.")
            logger.error("\n----- LLM 원본 응답 -----")
            logger.error(clean_response)
            logger.error("------------------------\n")
            
        # answer 값 정규화 및 검증
        valid_answer = answer.strip('"') if answer else "응답 파싱 실패"
        if valid_answer not in ["맞음", "아님"]:
            logger.error("\n===== 유효하지 않은 답변 =====")
            logger.error(f"원인: answer 값이 '맞음' 또는 '아님'이 아닙니다. (현재 값: {valid_answer})")
            logger.error("------------------------\n")
            valid_answer = "응답 파싱 실패"  # 파싱 실패로 처리하여 재시도 유도
            
        return {
            "answer": valid_answer,
            "rational": rational.strip('"') if rational else "응답 파싱 실패"
        }
    except Exception as e:
        logger.error("\n===== 파싱 실패 =====")
        logger.error(f"원인: {e}")
        logger.error("\n----- LLM 원본 응답 -----")
        logger.error(clean_response)
        logger.error("------------------------\n")
        
        return {
            "answer": "응답 파싱 실패",
            "rational": "응답 파싱 실패"
        }

def send_data_to_server2(data_items):
    """top-k 확률 정보를 포함하여 서버 2로 데이터 전송"""
    global last_transmission_time
    
    if not data_items:
        logger.warning("전송할 데이터가 없습니다.")
        return False
    
    logger.info(f"서버 2로 {len(data_items)}개 데이터(top-{TOP_K} 확률 포함) 전송 시도 중...")
    
    try:
        # 지연을 줄이기 위해 확률 데이터 압축
        compressed_items = []
        for item in data_items:
            # 기본 항목
            compressed_item = {
                "sentence": item.get("sentence", ""),
                "answer": item.get("answer", ""),
                "rational": item.get("rational", ""),
                "output": item.get("output", ""),
                "retry_count": item.get("retry_count", 0)
            }
            
            # top-k 토큰 정보 추가 (존재하는 경우)
            if "token_probs" in item:
                # 형식 최적화: [[{token_id, probability}, ...], [스텝2], ...]
                optimized_probs = []
                for step in item["token_probs"]:
                    step_data = []
                    for token_data in step[0]:  # 첫 번째 배치 항목만 사용
                        step_data.append({
                            "id": token_data["token_id"],
                            "p": round(token_data["probability"], 6)  # 소수점 6자리까지만 저장
                        })
                    optimized_probs.append(step_data)
                compressed_item["token_probs"] = optimized_probs
            
            compressed_items.append(compressed_item)
        
        response = requests.post(
            SERVER2_API,
            json=compressed_items,
            timeout=60  # 더 큰 데이터를 보내므로 타임아웃 증가
        )
        
        if response.status_code == 200:
            logger.info(f"서버 2 전송 성공: {response.json().get('message', '응답 메시지 없음')}")
            last_transmission_time = time.time()
            return True
        else:
            logger.error(f"서버 2 전송 실패: 상태 코드 {response.status_code}, 응답: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"서버 2 데이터 전송 중 오류: {e}")
        return False

def process_json_file(input_file, output_file):
    """JSON 파일의 문장을 처리하고 평가 결과를 출력 파일에 저장"""
    # 입력 파일 체크
    if not os.path.exists(input_file):
        logger.error(f"입력 파일이 존재하지 않습니다: {input_file}")
        return
    
    try:
        # 입력 파일 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 입력 데이터 형식 확인 및 문장 추출
        sentences = []
        if isinstance(data, list):
            # 데이터가 리스트인 경우
            if all(isinstance(item, str) for item in data):
                # 문자열 리스트인 경우
                sentences = data
                logger.info(f"입력 파일에서 {len(sentences)}개 문장(문자열 리스트) 로드됨")
            elif all(isinstance(item, dict) for item in data):
                # 딕셔너리 리스트인 경우
                if all("sentence" in item for item in data):
                    # 각 딕셔너리에 'sentence' 키가 있는 경우
                    sentences = [item["sentence"] for item in data]
                    logger.info(f"입력 파일에서 {len(sentences)}개 문장(딕셔너리의 'sentence' 필드) 로드됨")
                else:
                    # 다른 형식의 딕셔너리인 경우 첫 번째 값 사용
                    for item in data:
                        if item and isinstance(item, dict) and len(item) > 0:
                            first_key = next(iter(item))
                            sentences.append(item[first_key])
                    logger.info(f"입력 파일에서 {len(sentences)}개 문장(딕셔너리의 첫 번째 필드) 로드됨")
        elif isinstance(data, dict):
            # 데이터가 딕셔너리인 경우 
            if "sentences" in data and isinstance(data["sentences"], list):
                sentences = data["sentences"]
                logger.info(f"입력 파일에서 {len(sentences)}개 문장('sentences' 키) 로드됨")
            else:
                # 다른 형식의 딕셔너리인 경우 값들을 문장으로 사용
                sentences = list(data.values())
                logger.info(f"입력 파일에서 {len(sentences)}개 문장(딕셔너리 값) 로드됨")
        
        if not sentences:
            logger.error("입력 파일에서 처리할 문장을 찾을 수 없습니다.")
            return
        
        # 문자열이 아닌 항목 필터링
        sentences = [s for s in sentences if isinstance(s, str)]
        logger.info(f"처리할 문자열 문장: {len(sentences)}개")
        
    except Exception as e:
        logger.error(f"입력 파일을 불러오는 중 오류: {e}")
        return
    
    # 결과 파일 초기화 및 기존 결과 로드
    results = []
    processed_sentences = set()
    
    # 출력 파일이 존재하는 경우 기존 결과 로드
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 이미 처리된 문장들 기록
            for result in results:
                if 'sentence' in result:
                    processed_sentences.add(result['sentence'])
            
            logger.info(f"기존 결과 파일에서 {len(results)}개 항목 로드됨")
            logger.info(f"이미 처리된 문장: {len(processed_sentences)}개")
        except json.JSONDecodeError:
            logger.warning(f"결과 파일({output_file})이 비어있거나 올바른 JSON 형식이 아닙니다. 새로 시작합니다.")
            results = []
        except Exception as e:
            logger.warning(f"결과 파일({output_file})을 로드하는 중 오류: {e}. 새로 시작합니다.")
            results = []
    else:
        logger.info(f"결과 파일({output_file})이 존재하지 않습니다. 새로 생성합니다.")
        # 빈 파일 생성 - 디렉토리 확인
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"결과 파일 디렉토리 생성됨: {output_dir}")
        
        # 빈 결과 리스트 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            logger.info(f"빈 결과 파일이 생성되었습니다: {output_file}")
        except Exception as e:
            logger.error(f"결과 파일 생성 중 오류: {e}")
            return
    
    # 이제 processed_sentences와 sentences는 모두 문자열이므로 비교 가능
    remaining_sentences = [s for s in sentences if s not in processed_sentences]
    logger.info(f"처리할 남은 문장: {len(remaining_sentences)}개")
    
    if not remaining_sentences:
        logger.info("모든 문장이 이미 처리되었습니다.")
        return
    
    # 모델 관리자 인스턴스 가져오기
    global model_manager
    if not model_manager:
        logger.error("모델 관리자가 초기화되지 않았습니다.")
        return
    
    # 문장 처리 진행
    progress_bar = tqdm(remaining_sentences, desc="문장 처리 중")
    for sentence in progress_bar:
        # 결과가 이미 있는지 확인
        if sentence in processed_sentences:
            continue
        
        # 응답 생성할 프롬프트 준비
        prompt = create_main_prompt(sentence)
        
        # 타임스탬프 기록
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # 메인 모델로 응답 생성 (top-k 확률 포함)
            response_data = model_manager.generate_response(MAIN_MODEL, prompt)
            
            # 응답 결과
            response_text = response_data["text"]
            
            # 결과 분석
            answer, rational = None, None
            
            # Ollama 파싱 모델로 응답 파싱 시도
            parsing_prompt = f"""다음 텍스트에서 "answer"와 "rational" 값을 찾아 JSON 형식으로 반환해주세요:
            
            {response_text}
            
            다음 형식으로만 응답해주세요:
            {{"answer": "찾은 answer 값", "rational": "찾은 rational 값"}}"""
            
            try:
                parsed_data = model_manager.generate_response(PARSING_MODEL, parsing_prompt)
                parsed_text = parsed_data["text"].strip()
                
                # JSON 형식 확인 및 추출
                if "{" in parsed_text and "}" in parsed_text:
                    json_str = parsed_text[parsed_text.find("{"):parsed_text.rfind("}")+1]
                    parsed_json = json.loads(json_str)
                    answer = parsed_json.get("answer", "")
                    rational = parsed_json.get("rational", "")
                    logger.info(f"파싱 성공: answer='{answer}', rational 추출됨")
                else:
                    # 정규식으로 시도
                    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', parsed_text)
                    rational_match = re.search(r'"rational"\s*:\s*"([^"]+)"', parsed_text)
                    
                    if answer_match:
                        answer = answer_match.group(1)
                    if rational_match:
                        rational = rational_match.group(1)
                    
                    if answer or rational:
                        logger.info(f"정규식 파싱 성공: answer='{answer}', rational {'추출됨' if rational else '실패'}")
                    else:
                        logger.warning(f"파싱 실패: 형식에 맞지 않는 응답 - {parsed_text[:100]}...")
            except Exception as parse_error:
                logger.error(f"응답 파싱 오류: {parse_error}")
            
            # 결과 저장
            result_item = {
                "sentence": sentence,
                "prompt": prompt,
                "output": response_text,
                "answer": answer if answer else "",
                "rational": rational if rational else "",
                "timestamp": timestamp,
                "token_probs": response_data.get("token_probs", [])
            }
            
            # 결과 추가
            results.append(result_item)
            processed_sentences.add(sentence)
            
            # 큐에 전송할 데이터 추가
            data_queue.append(result_item)
            
            # 큐 크기가 임계값에 도달하면 전송
            if len(data_queue) >= QUEUE_SIZE_THRESHOLD:
                logger.info(f"큐 크기 임계값({QUEUE_SIZE_THRESHOLD}개) 도달, 서버 2로 전송 중...")
                if send_data_to_server2(data_queue):
                    data_queue.clear()
            
            # 시간 기반 전송 체크
            current_time = time.time()
            if data_queue and current_time - last_transmission_time > TRANSMISSION_INTERVAL:
                logger.info(f"마지막 전송 후 {TRANSMISSION_INTERVAL}초 경과, 서버 2로 전송 중...")
                if send_data_to_server2(data_queue):
                    data_queue.clear()
            
            # 5개 처리할 때마다 결과 저장
            if len(results) % 5 == 0:
                # 결과 파일에 저장
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                    logger.info(f"중간 결과 저장 완료: {len(results)}개 항목")
                except Exception as save_error:
                    logger.error(f"중간 결과 저장 오류: {save_error}")
        
        except Exception as e:
            logger.error(f"문장 처리 중 오류: {e}")
            # 오류 발생 시에도 저장 시도
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                logger.info(f"오류 후 결과 저장 완료: {len(results)}개 항목")
            except Exception as save_error:
                logger.error(f"오류 후 결과 저장 시도 중 추가 오류: {save_error}")
    
    # 마지막으로 큐에 남은 데이터 전송
    if data_queue:
        logger.info(f"남은 {len(data_queue)}개 데이터 서버 2로 전송 중...")
        send_data_to_server2(data_queue)
        data_queue.clear()
    
    # 최종 결과 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"최종 결과 저장 완료: {len(results)}개 항목")
    except Exception as e:
        logger.error(f"최종 결과 저장 오류: {e}")

if __name__ == "__main__":
    # 시작 로그
    logger.info("=" * 50)
    logger.info("데이터 생성기 시작")
    logger.info("=" * 50)
    
    # 패키지 설치 확인 메시지
    logger.info("필요 패키지: transformers, torch, bitsandbytes, numpy, tqdm, requests")
    
    # 서버 2 API 엔드포인트 설정
    SERVER2_API = input("서버 2 API 엔드포인트를 입력하세요 (기본값: http://192.168.0.1:5000/receive_data): ") or "http://192.168.10.4:5000/receive_data"
    logger.info(f"서버 2 API 엔드포인트: {SERVER2_API}")
    
    # 모델 관리자 초기화 (전역 변수)
    model_manager = ModelManager()
    
    # 파일 경로 설정
    input_file = input("입력 JSON 파일 경로를 입력하세요 (기본값: input.json): ") or "input.json"
    output_file = input("출력 JSON 파일 경로를 입력하세요 (기본값: output.json): ") or "output.json"
    logger.info(f"입력 파일: {input_file}, 출력 파일: {output_file}")
    
    # GPU 확인 및 설정
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"사용 가능한 GPU: {device_count}개")
        for i in range(device_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.1f} GB")
        
        # 모델 배치 안내
        if device_count >= 2:
            logger.info("\n다중 GPU 환경이 감지되었습니다.")
            logger.info(f"- 32B 모델({MAIN_MODEL})은 더 큰 VRAM의 GPU에 자동 배치됩니다.")
            logger.info(f"- 8B 모델({PARSING_MODEL})은 다른 GPU에 자동 배치됩니다.")
            logger.info("- 이 배치는 'device_map=\"auto\"' 설정으로 자동 결정됩니다.")
    else:
        logger.warning("경고: GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다 (매우 느릴 수 있음).")
    
    # 출력 디렉토리 및 파일 체크
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"출력 디렉토리 생성됨: {output_dir}")
        except Exception as e:
            logger.error(f"출력 디렉토리 생성 실패: {e}")
    
    # 파일 처리
    process_json_file(input_file, output_file)
    
    # 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("=" * 50)
    logger.info("데이터 생성기 종료")
    logger.info("=" * 50)
