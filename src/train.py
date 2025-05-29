import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import numpy as np
import os
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# 상수 정의
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BASE_PATH = "./model/base"
TUNED_PATH = "./model/tuned"
DATA_PATH = "./output_txt.json"
FIGURE_PATH = "./training_loss.jpg"

# 구분선 함수 정의
def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

# 장치 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0번 GPU만 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_section("환경 설정")
print(f"사용 장치: {device}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
    print(f"가용 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 데이터 로드
print_section("데이터 로드")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"원본 데이터 항목 수: {len(data)}")
print(f"첫 번째 데이터 샘플: {data[0]}")

# 데이터 전처리
train_data = []
for item in data:
    # sentence 입력으로, output을 타겟으로 사용
    train_data.append({
        "input": item["sentence"],
        "output": item["output"]
    })

# 학습/검증 데이터 분할 (8:2)
np.random.shuffle(train_data)
split_idx = int(len(train_data) * 0.8)
train_dataset = Dataset.from_list(train_data[:split_idx])
eval_dataset = Dataset.from_list(train_data[split_idx:])

print(f"학습 데이터 크기: {len(train_dataset)}")
print(f"검증 데이터 크기: {len(eval_dataset)}")
print("\n학습 데이터 샘플:")
for i in range(min(3, len(train_dataset))):
    print(f"  {i+1}. 입력: {train_dataset[i]['input']}")
    print(f"     출력: {train_dataset[i]['output']}\n")

# 모델 로드 및 LoRA 설정 부분 수정
print_section("모델 및 토크나이저 로드")
print(f"기본 모델: {MODEL_NAME}")

# base 디렉토리 확인 및 모델 다운로드
import os

# base 디렉토리가 없으면 생성
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# base 디렉토리가 비어있는지 확인
is_base_empty = len(os.listdir(BASE_PATH)) == 0

# 토크나이저 로드
if is_base_empty:
    print(f"베이스 모델 디렉토리({BASE_PATH})가 비어 있습니다. Hugging Face에서 모델을 다운로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(BASE_PATH)
    print(f"토크나이저를 {BASE_PATH}에 저장했습니다.")
else:
    print(f"{BASE_PATH}에서 토크나이저를 로드합니다.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

# <think>, </think> 태그를 토크나이저에 추가
special_tokens_dict = {
    'additional_special_tokens': ['<think>', '</think>']
}
tokenizer.add_special_tokens(special_tokens_dict)
print(f"특수 토큰 추가: {special_tokens_dict['additional_special_tokens']}")
print(f"토크나이저 어휘 크기: {len(tokenizer)}")

# 모델 로드
if is_base_empty:
    print(f"Hugging Face에서 베이스 모델을 다운로드하여 {BASE_PATH}에 저장합니다...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    base_model.save_pretrained(BASE_PATH)
    print(f"모델을 {BASE_PATH}에 저장했습니다.")
    model = base_model
else:
    print(f"{BASE_PATH}에서 기본 모델을 로드합니다.")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

print(f"모델 타입: {model.__class__.__name__}")
print(f"모델 데이터 타입: {model.dtype}")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# LoRA 설정
print_section("LoRA 설정")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
print(f"LoRA 구성: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
print(f"타겟 모듈: {lora_config.target_modules}")
print(f"LoRA 드롭아웃: {lora_config.lora_dropout}")

# 모델을 LoRA 학습을 위해 준비
print("모델을 LoRA 학습 용도로 변환 중...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 출력
print("\n학습 가능한 파라미터 정보:")
model.print_trainable_parameters()

# 모델의 임베딩 크기 조정
model.resize_token_embeddings(len(tokenizer))
print(f"토크나이저에 맞게 임베딩 크기 조정: {len(tokenizer)}")

# 토크나이저 설정
tokenizer.pad_token = tokenizer.eos_token
print(f"패딩 토큰 설정: '{tokenizer.pad_token}'")

# 데이터 전처리 함수 개선 - 간결한 프롬프트 사용
def preprocess_function(examples):
    prompts = []
    for inp in examples["input"]:
        prompt = f"<s>[INST] 다음 영어 문장의 문법을 평가하세요.\n" + \
                 f"<think>태그 안에 분석 과정을 작성하고, 맞음/틀림을 판단하세요.\n" + \
                 f"영어 문장: {inp} [/INST]"
        prompts.append(prompt)
    
    targets = []
    for out in examples["output"]:
        targets.append(" " + out + "</s>")
    
    # 입력과 출력을 합쳐서 한 번에 토큰화
    combined_texts = []
    for p, t in zip(prompts, targets):
        combined_texts.append(p + t)
    
    result = tokenizer(combined_texts, truncation=True, padding="max_length", max_length=512)
    result["labels"] = result["input_ids"].copy()
    
    return result

# 데이터셋 전처리
print_section("데이터셋 전처리")
print("학습 데이터 전처리 중...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
print("검증 데이터 전처리 중...")
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

print(f"전처리된 학습 데이터 형태: {train_dataset[0].keys()}")
print(f"입력 길이: {len(train_dataset[0]['input_ids'])}")
print(f"라벨 길이: {len(train_dataset[0]['labels'])}")

# 학습 인자 설정 수정
print_section("학습 설정")
training_args = TrainingArguments(
    output_dir=TUNED_PATH,
    num_train_epochs=4,  # 에폭 수 증가
    per_device_train_batch_size=2,  # 배치 크기 조정
    per_device_eval_batch_size=2,
    warmup_steps=200,
    weight_decay=0.02,  # 규제 강화
    logging_dir="./logs",
    logging_steps=10,  # 로깅 빈도 증가
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    learning_rate=5e-5,  # 학습률 조정
    bf16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    gradient_accumulation_steps=8,  # 그래디언트 누적 단계 증가
    report_to="none",
    # fp16=True,  # 필요시 fp16 활성화
)

print(f"학습 에폭: {training_args.num_train_epochs}")
print(f"배치 크기: {training_args.per_device_train_batch_size} (장치당)")
print(f"그래디언트 누적 단계: {training_args.gradient_accumulation_steps}")
print(f"유효 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"학습률: {training_args.learning_rate}")
print(f"데이터 타입: {'bfloat16' if training_args.bf16 else 'float32'}")
print(f"저장 경로: {training_args.output_dir}")

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 모델 학습
print_section("모델 학습")
print("학습 시작...")
train_result = trainer.train()

print("\n학습 결과:")
print(f"총 학습 단계: {train_result.global_step}")
print(f"학습 손실: {train_result.training_loss:.4f}")
print(f"학습 시간: {train_result.metrics['train_runtime']:.2f}초")
print(f"학습 샘플/초: {train_result.metrics['train_samples_per_second']:.2f}")

# 학습 완료 후 LoRA 가중치를 기본 모델에 병합
print_section("모델 병합 및 저장")
print("LoRA 가중치를 기본 모델에 병합합니다...")
merged_model = model.merge_and_unload()

# 병합된 전체 모델 저장
print(f"병합된 모델 저장 중... 경로: {TUNED_PATH}")
merged_model.save_pretrained(TUNED_PATH)
tokenizer.save_pretrained(TUNED_PATH)
print(f"병합된 전체 모델이 {TUNED_PATH}에 저장되었습니다.")
print(f"모델 크기: {sum(p.numel() for p in merged_model.parameters()) / 1e6:.2f}M 파라미터")

# 학습 및 검증 손실 그래프 생성
print_section("학습 손실 그래프 생성")
train_loss = trainer.state.log_history
train_loss_values = []
eval_loss_values = []
steps = []

for entry in train_loss:
    if "loss" in entry:
        train_loss_values.append(entry["loss"])
        steps.append(entry["step"])
    if "eval_loss" in entry:
        eval_loss_values.append(entry["eval_loss"])

print(f"기록된 학습 손실 포인트: {len(train_loss_values)}")
print(f"기록된 검증 손실 포인트: {len(eval_loss_values)}")

plt.figure(figsize=(10, 6))
plt.plot(steps[:len(train_loss_values)], train_loss_values, label="Training Loss")
if eval_loss_values:
    eval_steps = [step for i, step in enumerate(steps) if i % (len(steps) // len(eval_loss_values)) == 0][:len(eval_loss_values)]
    plt.plot(eval_steps, eval_loss_values, label="Validation Loss")
    
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(FIGURE_PATH)
print(f"학습 그래프가 {FIGURE_PATH}에 저장되었습니다.")

# 학습된 모델 테스트
print_section("학습된 모델 테스트")
test_sentences = [
    "The kids are playing in the park.",
    "The kids is talking a movie currently.",
    "She have been working here for five years."
]

# 병합된 모델 직접 로드
print("병합된 모델 로드 중...")
trained_model = AutoModelForCausalLM.from_pretrained(
    TUNED_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
trained_model.eval()
print("모델 로드 완료!")

for i, test_sentence in enumerate(test_sentences):
    print(f"\n테스트 #{i+1}: '{test_sentence}'")
    prompt = f"<s>[INST] 다음 영어 문장의 문법을 평가하세요.\n" + \
             f"<think>태그 안에 분석 과정을 작성하고, 맞음/틀림을 판단하세요.\n" + \
             f"영어 문장: {test_sentence} [/INST]"
    
    print(f"프롬프트: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("응답 생성 중...")
    with torch.no_grad():
        outputs = trained_model.generate(
            **inputs,
            max_new_tokens=256,  # 출력 길이 축소
            num_return_sequences=1,
            temperature=0.1,  
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 응답에서 프롬프트 부분 제거하기
    response_only = response.split("[/INST]")[1].strip() if "[/INST]" in response else response
    
    print(f"\n[입력] {test_sentence}")
    print(f"[출력] {response_only}")
    
    # 입력과 출력 사이에 구분선 추가
    if i < len(test_sentences) - 1:
        print("\n" + "-"*40)

print_section("실행 완료")
print("모든 과정이 성공적으로 완료되었습니다!")
