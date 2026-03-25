# MediaTek LLaMA Runner 코드 흐름 상세 분석서

> **가정 (예시 값)**
> - 모델: LLaMA 3 8B Instruct (MediaTek NPU, A16W4 양자화)
> - 프롬프트: `"What is the meaning of life?"` → 토큰화 결과 9개 토큰
> - 토큰 ID: `[128000, 3923, 374, 279, 7438, 315, 2324, 30, 128001]` (BOS 포함)
> - vocab_size: 128000, hidden_size: 4096, num_head: 32, num_layer: 32
> - prompt_token_batch_size: 128, cache_size: 512, max_token_length: 8192
> - 모든 데이터 타입: FP32
> - 모델 청크: 4개 (8레이어 × 4 = 32레이어)
> - max_response: 50 토큰

---

## 목차

1. [전체 아키텍처 개요 — Meta(XNNPACK) vs MediaTek(NPU)](#1-전체-아키텍처-개요)
2. [클래스 계층 구조](#2-클래스-계층-구조)
3. [main() 진입점](#3-main-진입점)
4. [모델 옵션 및 경로 구성](#4-모델-옵션-및-경로-구성)
5. [LlamaRuntime::Initialize — 핵심 초기화](#5-llamaruntimeinitialize--핵심-초기화)
6. [Inference 전체 흐름](#6-inference-전체-흐름)
7. [Prefill 단계 — digest_prompt()](#7-prefill-단계--digest_prompt)
8. [LlamaRuntime::Run — 단일 Forward Pass](#8-llamaruntimerun--단일-forward-pass)
9. [모델 스왑 — SwapModel()](#9-모델-스왑--swapmodel)
10. [Generation 단계 — gen_response()](#10-generation-단계--gen_response)
11. [헬퍼 라이브러리 상세 분석](#11-헬퍼-라이브러리-상세-분석)
12. [Model-Specific vs 공용 부분 구분](#12-model-specific-vs-공용-부분-구분)
13. [신규 모델 추가 가이드](#13-신규-모델-추가-가이드)
14. [Meta TextLLMRunner와의 상세 비교](#14-meta-textllmrunner와의-상세-비교)
15. [전체 호출 스택 요약](#15-전체-호출-스택-요약)

---

## 1. 전체 아키텍처 개요

### Meta(XNNPACK) Runner vs MediaTek(NPU) Runner

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Meta TextLLMRunner (XNNPACK)                        │
│                                                                        │
│   main.cpp → TextLLMRunner → Module::execute("forward")               │
│                                    │                                   │
│                              단일 .pte 파일                            │
│                              XNNPACK delegate                          │
│                              CPU에서 실행                              │
│                              동적 shape 지원                           │
│                              내부 KV 캐시 관리                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                  MediaTek MTKLlamaRunner (NPU)                         │
│                                                                        │
│   main.cpp → MTKLlamaRunner → LlamaRuntime                            │
│                                    │                                   │
│                              8개 .pte 파일 (4 prompt + 4 gen)         │
│                              NeuronBufferAllocator (NPU 메모리)       │
│                              NPU에서 실행                              │
│                              고정 shape (배치 패딩 필요)              │
│                              외부 KV 캐시 관리 (수동)                 │
│                              외부 Token Embedding (파일 LUT)          │
│                              외부 Rotary Embedding (수동 생성)        │
│                              외부 Attention Mask (수동 빌드)          │
│                              모델 청크 분할 (4개)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### MediaTek이 자체 Runner를 만든 이유

| 특성 | Meta TextLLMRunner | MediaTek MTKLlamaRunner |
|------|-------------------|------------------------|
| 실행 장치 | CPU (XNNPACK) | MediaTek NPU (Neuron) |
| 모델 파일 | 단일 .pte | 8개 .pte (prompt 4 + gen 4) |
| 입력 shape | 동적 (1~2048) | 고정 (128 또는 1) |
| KV 캐시 | 모델 내부 관리 | 외부 수동 관리 |
| Token Embedding | 모델 내부 | 외부 파일 LUT (embedding.bin) |
| Rotary Embedding | 모델 내부 | 외부 수동 계산 |
| Attention Mask | 모델 내부 | 외부 수동 빌드 |
| 메모리 할당 | 표준 malloc | NeuronBufferAllocator (NPU 전용) |
| 모델 스왑 | 불필요 | prompt↔gen 모델 핫스왑 필요 |
| 샘플링 | temperature+top-p | argmax only (greedy) |

---

## 2. 클래스 계층 구조

```
IRunner (인터페이스, extension/llm/runner/irunner.h)
  └── MTKLlamaRunner (examples/mediatek/executor_runner/mtk_llama_runner.h)
        ├── owns: LlamaRuntime
        │     ├── owns: TokenEmbeddingLut (embedding.bin 로드)
        │     ├── owns: RotaryEmbeddingMasterLut (RoPE 테이블)
        │     └── owns: vector<ModelChunk> (4개 디코더 청크)
        │           └── LlamaModelChunk : ModelChunk
        │                 ├── owns: MaskBuilder (어텐션 마스크 생성)
        │                 ├── uses: RotaryEmbeddingMasterLut* (비소유)
        │                 └── inherits: MultiModelLoader<size_t>
        │                       └── manages: prompt/gen .pte 모델 인스턴스
        └── owns: Tokenizer (TikToken)

파일 구조:
examples/mediatek/executor_runner/
├── mtk_llama_executor_runner.cpp  ← main() 독립 실행기
├── mtk_llama_runner.h/cpp         ← IRunner 구현체
└── llama_runner/
    ├── LlamaConfig.h              ← LlamaModelOptions, LlamaModelPaths
    ├── LlamaRuntime.h/cpp         ← 런타임 오케스트레이터
    ├── LlamaModelChunk.h/cpp      ← LLaMA 전용 모델 청크
    ├── ModelChunk.h/cpp            ← 범용 모델 청크 (재사용 가능)
    ├── MultiModelLoader.h/cpp      ← 멀티모델 로더 템플릿 (재사용 가능)
    ├── Utils.h                     ← 유틸리티 (Timer, argmax, split 등)
    ├── FileMemMapper.h             ← 파일 메모리 매핑 (재사용 가능)
    └── llm_helper/
        ├── include/
        │   ├── llm_types.h             ← LLM 데이터 타입 정의
        │   ├── llama_runner_values.h   ← 하드코딩된 모델 상수값
        │   ├── token_embedding.h       ← 토큰 임베딩 LUT
        │   ├── rotary_embedding.h      ← RoPE 테이블 생성
        │   └── mask_builder.h          ← 어텐션 마스크 빌더
        └── (구현 .cpp 파일들)
```

---

## 3. main() 진입점

**파일**: `examples/mediatek/executor_runner/mtk_llama_executor_runner.cpp`

### 3.1 커맨드라인 인자

```
실행 예시 (Android 디바이스):
  mtk_llama_executor_runner \
    --prompt_token_batch_size=128 \
    --cache_size=512 \
    --hidden_size=4096 \
    --num_head=32 \
    --num_layer=32 \
    --max_token_length=8192 \
    --rot_emb_base=500000 \
    --input_type=fp32 --output_type=fp32 --cache_type=fp32 \
    --mask_type=fp32 --rot_emb_type=fp32 \
    --tokenizer_path=/data/local/tmp/et-mtk/llama3/tokenizer.model \
    --tokenizer_type=tiktoken \
    --token_embedding_path=/.../embedding_llama3-8B-instruct_fp32.bin \
    --prompt_model_paths=/.../128t512c_0.pte,/.../128t512c_1.pte,/.../128t512c_2.pte,/.../128t512c_3.pte \
    --gen_model_paths=/.../1t512c_0.pte,/.../1t512c_1.pte,/.../1t512c_2.pte,/.../1t512c_3.pte \
    --max_response=50 \
    --prompt_file=/data/local/tmp/prompt.txt
```

주요 플래그:

| 카테고리 | 플래그 | 기본값 | 설명 |
|---------|--------|--------|------|
| **모델 크기** | `prompt_token_batch_size` | 128 | Prefill 배치 크기 |
| | `cache_size` | 1024 | KV 캐시 길이 |
| | `hidden_size` | 4096 | 히든 차원 |
| | `num_head` | 32 | 어텐션 헤드 수 |
| | `num_layer` | 32 | 트랜스포머 레이어 수 |
| | `max_token_length` | 2048 | 최대 지원 토큰 길이 |
| | `rot_emb_base` | 10000 | RoPE theta |
| **데이터 타입** | `input_type` | int16 | 모델 입력 타입 |
| | `output_type` | int16 | 모델 출력 타입 |
| | `cache_type` | int16 | KV 캐시 타입 |
| **경로** | `prompt_model_paths` | | Prefill 모델 (쉼표 구분) |
| | `gen_model_paths` | | Generation 모델 (쉼표 구분) |
| | `token_embedding_path` | embedding.bin | 임베딩 LUT 파일 |
| **토크나이저** | `tokenizer_type` | tiktoken | bpe / tiktoken / hf |
| | `vocab_size` | 128000 | 어휘 크기 |
| **추론** | `max_response` | 50 | 최대 생성 토큰 수 |

### 3.2 main() 흐름

```cpp
int main() {
  runtime_init();                           // ExecuTorch 런타임 초기화
  gflags::ParseCommandLineFlags();          // 인자 파싱

  LlamaModelOptions options = get_model_options(); // 플래그 → 구조체
  LlamaModelPaths paths = get_model_paths();       // 플래그 → 구조체

  // prompt 모델이 없으면 배치 크기를 1로 (gen 모델만 사용)
  if (paths.prompt_model_paths.empty())
    options.prompt_token_batch_size = 1;

  LlamaRuntime llama_runtime;

  // [1] 모델 초기화
  auto tokenizer = load_tokenizer();          // TikToken/BPE/HF 로드
  llama_runtime.Initialize(options, paths);   // NPU 모델 로드 + 헬퍼 초기화

  // [2] 추론 실행
  std::string prompt = read_file(FLAGS_prompt_file);
  inference(llama_runtime, tokenizer, prompt);

  // [3] 모델 해제
  llama_runtime.Release();
}
```

---

## 4. 모델 옵션 및 경로 구성

### 4.1 LlamaModelOptions

```cpp
// llama_runner_values.h의 하드코딩 값 (MTKLlamaRunner에서 사용)
LlamaModelOptions options = {
  .prompt_token_batch_size = 128,   // Prefill: 128 토큰 배치
  .cache_size = 512,                // KV 캐시 512 위치
  .hidden_size = 4096,              // LLaMA 3 8B 히든 크기
  .num_head = 32,                   // 32개 어텐션 헤드
  .num_layer = 32,                  // 32개 트랜스포머 레이어
  .head_dim = 0,                    // 0이면 자동계산: 4096/32 = 128
  .max_token_length = 8192,         // 최대 8192 토큰
  .rot_emb_base = 500000,           // LLaMA 3의 RoPE theta
  .model_input_type = FP32,         // 입출력 모두 FP32
  .model_output_type = FP32,
  .cache_type = FP32,
  .mask_type = FP32,
  .rot_emb_type = FP32
};
```

### 4.2 LlamaModelPaths

```
파일 구조 (Android 디바이스):
/data/local/tmp/et-mtk/llama3/
├── tokenizer.model                            ← 토크나이저
├── embedding_llama3-8B-instruct_fp32.bin     ← 토큰 임베딩 LUT (4.1GB)
│                                                vocab_size(128000) × hidden_size(4096) × 4bytes
│
├── llama3-8B-instruct_A16W4_4_chunks_128t512c_0.pte  ← Prompt 모델 청크 0 (레이어 0~7)
├── llama3-8B-instruct_A16W4_4_chunks_128t512c_1.pte  ← Prompt 모델 청크 1 (레이어 8~15)
├── llama3-8B-instruct_A16W4_4_chunks_128t512c_2.pte  ← Prompt 모델 청크 2 (레이어 16~23)
├── llama3-8B-instruct_A16W4_4_chunks_128t512c_3.pte  ← Prompt 모델 청크 3 (레이어 24~31)
│
├── llama3-8B-instruct_A16W4_4_chunks_1t512c_0.pte    ← Gen 모델 청크 0
├── llama3-8B-instruct_A16W4_4_chunks_1t512c_1.pte    ← Gen 모델 청크 1
├── llama3-8B-instruct_A16W4_4_chunks_1t512c_2.pte    ← Gen 모델 청크 2
└── llama3-8B-instruct_A16W4_4_chunks_1t512c_3.pte    ← Gen 모델 청크 3

파일명 해석:
  llama3-8B-instruct_A16W4_4_chunks_128t512c_0.pte
  └── 모델명 ──────── 양자화 청크수 ─ 토큰/캐시 ─ 청크ID
                       A16=Act16bit    128t = 128 토큰 배치
                       W4=Weight4bit   512c = 512 캐시 크기
```

**왜 8개 .pte 파일인가?**

1. **Prompt vs Gen 모델 분리**: Prompt 모델은 입력 shape [128, 4096], Gen 모델은 [1, 4096]
   - NPU는 고정 shape 모델만 지원 → 배치 크기별 별도 컴파일 필요
2. **4개 청크 분할**: 32 레이어를 8 레이어씩 4개로 분할
   - NPU 메모리 한계 → 전체 모델이 한 번에 로드 불가
   - 청크별 순차 실행으로 메모리 재사용

---

## 5. LlamaRuntime::Initialize — 핵심 초기화

**파일**: `llama_runner/LlamaRuntime.cpp:23`

```
LlamaRuntime::Initialize(modelOptions, modelPaths)
│
├── [1] Rotary Embedding 마스터 LUT 생성
│   headDim = 4096 / 32 = 128
│   rotEmbDim = 128 * 1.0 = 128
│   RotaryEmbeddingMasterLut(FP32, max=8192, dim=128, base=500000)
│   → generate(): 8192개 위치 × 128차원 × cos/sin = 8MB 테이블 생성
│
├── [2] 모델 청크 수 결정
│   numChunk = gen_model_paths.size() = 4
│   numCache = 2 × 32 / 4 = 16 (청크당 K캐시 8개 + V캐시 8개)
│   initBatchSize = 128 (prompt 모델 사용)
│
├── [3] 4개 LlamaModelChunk 생성
│   for chunkIdx in 0..3:
│     modelPathMap = {
│       128: "...128t512c_{chunkIdx}.pte",  // Prompt 모델
│       1:   "...1t512c_{chunkIdx}.pte"     // Gen 모델
│     }
│     LlamaModelChunk(modelPathMap, options, initBatch=128,
│                      numCache=16, numRotEmb=1, enableSWA=false,
│                      rotEmbMasterLut)
│
├── [4] 청크 간 버퍼 연결 & 초기화
│   chunk[0]: 독립 입력 버퍼 (토큰 임베딩에서 연결)
│   chunk[1].SetInputBuffer(chunk[0].GetOutputBuffer()) ← 출력→입력 연결
│   chunk[2].SetInputBuffer(chunk[1].GetOutputBuffer())
│   chunk[3].SetInputBuffer(chunk[2].GetOutputBuffer())
│   각 chunk.Initialize() → .pte 로드, NPU 메모리 할당, IO 버퍼 설정
│
├── [5] Token Embedding LUT 로드
│   TokenEmbeddingLut("embedding.bin", FP32, hidden_size=4096)
│   → mmap으로 embedding.bin 매핑 (128000 × 4096 × 4 = 2GB)
│   → vocab_size = filesize / (4096 × 4) = 128000
│
└── [6] Token Embedding 출력을 첫 번째 청크 입력에 연결
    tokenEmbInput = chunk[0].GetInputBuffer()
    tokenEmbLut.setOutput(tokenEmbInput.data, tokenEmbInput.nbytes)
```

### 초기화 후 메모리 레이아웃

```
                    Token Embedding LUT (mmap, 2GB)
                              │
                              ▼ lookupEmbedding()
┌───────────────────────────────────────────────────────────────────┐
│ Chunk 0 (Layer 0~7)                                              │
│  Input: [128, 4096] FP32 ← TokenEmbLut 출력                     │
│  Mask:  [128, 512+128] FP32 ← MaskBuilder                       │
│  RotEmb: [128, 128] FP32 ← RotEmbMasterLut                      │
│  KV Cache × 16: [32, 128, 512] FP32 (8K + 8V)                   │
│  Output: [128, 4096] FP32                                        │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │ output → input 연결
┌─────────────────────────────────▼─────────────────────────────────┐
│ Chunk 1 (Layer 8~15)                                             │
│  Input: [128, 4096] FP32 ← Chunk 0 출력                         │
│  ... (동일 구조)                                                  │
│  Output: [128, 4096] FP32                                        │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ Chunk 2 (Layer 16~23)                                            │
│  ... (동일 구조)                                                  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ Chunk 3 (Layer 24~31)                                            │
│  ... (동일 구조)                                                  │
│  Output: [128, 128000] FP32 ← Logits!                            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 6. Inference 전체 흐름

**파일**: `mtk_llama_runner.cpp:270` (또는 `mtk_llama_executor_runner.cpp:271`)

```
inference(llama_runtime, tokenizer, prompt)
│
├── [1] 프롬프트 토큰화
│   tokenizer->encode("What is the meaning of life?", bos=1, eos=0)
│   → [128000, 3923, 374, 279, 7438, 315, 2324, 30]
│   (BOS=128000 포함, 8개 토큰)
│
├── [2] Prefill (digest_prompt)
│   digest_prompt(llama_runtime, tokenizer, input_tokens)
│   → 8개 토큰을 128 배치로 패딩하여 1회 실행
│   → logits에서 argmax → first_output_token (예: 578 = "The")
│
└── [3] Generation (gen_response)
    gen_response(llama_runtime, tokenizer, first_output_token=578)
    → 모델을 Gen 모드로 스왑 (128→1 배치)
    → 자기회귀 루프: 최대 50회 반복
       Run({578}) → logits → argmax → 7438 → decode → " meaning"
       Run({7438}) → logits → argmax → ... → decode → 출력
       ...
    → EOS(128001) 또는 max_response(50) 도달 시 종료
```

---

## 7. Prefill 단계 — digest_prompt()

**파일**: `mtk_llama_runner.cpp:146` (또는 `mtk_llama_executor_runner.cpp:147`)

```
digest_prompt(llama_runtime, tokenizer, [128000, 3923, 374, 279, 7438, 315, 2324, 30])
│
├── input_token_count = 8
├── prompt_token_batch_size = 128 (현재 배치 크기)
│
├── [배치 분할 로직 — getNextTokens()]
│   num_tok_remain = 8
│   remainder = 8 % 128 = 8
│   num_new_tokens = 8 (나머지가 0이 아니므로 remainder 사용)
│   → next_tokens = [128000, 3923, 374, 279, 7438, 315, 2324, 30] (8개)
│
│   만약 프롬프트가 200 토큰이었다면:
│     1회차: remainder=200%128=72 → 72개 토큰 처리
│     2회차: remainder=128%128=0 → 128개 토큰 처리
│     ※ 뒤에서부터 배치 단위로 나눔 (나머지를 먼저 처리)
│
├── [Forward Pass 실행]
│   timer_digest_prompt.Start()
│   while (0 < 8):
│     next_tokens = [128000, 3923, 374, 279, 7438, 315, 2324, 30]
│     logits = llama_runtime.Run(next_tokens)  ← ★ 핵심 실행
│     cur_token_index += 8
│   timer_digest_prompt.End()
│   → "Done analyzing prompt in 0.15 sec (853.3 tok/s)"
│     (128 ideal tokens / 0.15 sec, 패딩 포함)
│
└── [Argmax 샘플링]
    vocab_size = 128000
    logits_type = FP32
    first_output_token = argmax<float>(logits, 128000)
    → logits[128000] 배열에서 최대값 인덱스 → 578 ("The")

반환: 578
```

### Prefill 시 배치 분할 패턴

```
예시 1: 프롬프트 8 토큰, 배치 128
  → 1회 실행: 8 토큰 + 120 패딩 = 128

예시 2: 프롬프트 200 토큰, 배치 128
  → 1회차: 72 토큰 + 56 패딩 = 128 (remainder 먼저)
  → 2회차: 128 토큰 + 0 패딩 = 128

예시 3: 프롬프트 256 토큰, 배치 128
  → 1회차: 128 토큰 = 128 (정확히 배치 크기)
  → 2회차: 128 토큰 = 128

예시 4: 프롬프트 300 토큰, 배치 128
  → 1회차: 44 토큰 + 84 패딩 = 128
  → 2회차: 128 토큰 = 128
  → 3회차: 128 토큰 = 128
```

---

## 8. LlamaRuntime::Run — 단일 Forward Pass

**파일**: `llama_runner/LlamaRuntime.cpp:149`

```
LlamaRuntime::Run([128000, 3923, 374, 279, 7438, 315, 2324, 30])
│
├── [1] 패딩 계산
│   tokenIndex = 0 (첫 실행)
│   numNewInputToken = 8
│   mTokenBatchSize = 128
│   padSize = 128 - 8 = 120
│   isLeftPadAllowed = (tokenIndex == 0) → true (캐시가 비어있음)
│
│   Left-padding:
│     curInputTokens = [0,0,...(120개)...,0, 128000, 3923, 374, 279, 7438, 315, 2324, 30]
│     총 128개 토큰 (120 패딩 + 8 실제)
│
│   ※ Left-padding이 가능한 이유: 캐시가 비어있어 왼쪽에 쓰레기 값이 없음
│   ※ 이후 실행에서는 Right-padding 사용 (왼쪽 캐시가 이미 채워져 있으므로)
│
├── [2] Token Embedding Lookup
│   mTokenEmbLut->lookupEmbedding(curInputTokens)
│   각 토큰 ID → embedding.bin에서 4096차원 벡터 로드
│   결과: [128, 4096] FP32 텐서 (Chunk 0 입력 버퍼에 직접 기록)
│
│   예시:
│     token 0 (패딩) → embedding[0]: [0.01, -0.03, 0.15, ...]  (4096개 float)
│     token 128000 (BOS) → embedding[128000]: [-0.5, 0.3, ...]
│     token 3923 ("What") → embedding[3923]: [0.12, -0.45, ...]
│     ...
│
├── [3] 4개 디코더 청크 순차 실행
│   for chunk in [chunk0, chunk1, chunk2, chunk3]:
│
│     [3a] 패딩 설정
│       chunk.SetLeftPadding(120)
│       → MaskBuilder가 120개 패딩 위치를 마스킹
│       → RotaryEmbedding이 실제 토큰 위치(0~7)에만 적용
│       → 캐시 저장 시 패딩 위치 무시
│
│     [3b] 모델 실행
│       chunk.Run()
│       내부:
│         (a) UpdatePosEmbAndMask(): RoPE 임베딩 설정 + 마스크 빌드
│         (b) PrepareCacheIOs(): KV 캐시 입출력 버퍼 설정
│         (c) SetBackendInputs/Outputs(): NPU 버퍼 바인딩
│         (d) GetModelMethod().execute(): ★ NPU 실행
│         (e) PaddingPostprocess(): 패딩 캐시 롤백
│         (f) AdvanceTokenIndex(): 토큰 인덱스 += 8
│
│   Chunk 0: [128, 4096] → NPU 실행(Layer 0~7) → [128, 4096]
│   Chunk 1: [128, 4096] → NPU 실행(Layer 8~15) → [128, 4096]
│   Chunk 2: [128, 4096] → NPU 실행(Layer 16~23) → [128, 4096]
│   Chunk 3: [128, 4096] → NPU 실행(Layer 24~31) → [128, 128000] (logits)
│
├── [4] 토큰 인덱스 업데이트
│   mTokenIndex += 8 → mTokenIndex = 8
│
└── [5] Logits 반환
    finalChunk.GetOutputBuffer() → [128, 128000] FP32
    lastLogits=true, mTokenBatchSize=128:
      rightPadSize = 0 (left padding이었으므로)
      offset = (logitsSize/128) * (128 - 1 - 0)
             = (128000*4) * 127  ← 마지막 실제 토큰(position 127)의 logits
    return logitsData + offset → [128000] FP32 포인터
```

### Left-padding vs Right-padding 시각화

```
Left-padding (첫 실행, 캐시 비어있음):
  Input:  [PAD PAD ... PAD | BOS "What" "is" "the" "meaning" "of" "life" "?"]
  Cache:  [--- --- ... --- |  0    1    2    3      4       5     6    7  ]
                             캐시 위치 0~7에 실제 데이터 저장
                             패딩 위치의 캐시는 롤백(삭제)

Right-padding (이후 실행, 캐시에 이전 데이터 있음):
  Input:  [token | PAD PAD ... PAD]
  Cache:  [  8   | --- --- ... ---]
           위치 8에 저장     패딩 캐시 롤백
```

---

## 9. 모델 스왑 — SwapModel()

**파일**: `llama_runner/LlamaRuntime.cpp:125`

```
LlamaRuntime::SwapModel(batchSize=1)
│
├── [목적] Prompt 모델(128 배치) → Gen 모델(1 배치)로 교체
│
├── [멀티스레딩 스왑]
│   4개 청크를 병렬로 핫스왑:
│     Thread 0: chunk[0].HotSwapModel(1) → 128t512c_0.pte → 1t512c_0.pte
│     Thread 1: chunk[1].HotSwapModel(1) → 128t512c_1.pte → 1t512c_1.pte
│     Thread 2: chunk[2].HotSwapModel(1) → 128t512c_2.pte → 1t512c_2.pte
│     Thread 3: chunk[3].HotSwapModel(1) → 128t512c_3.pte → 1t512c_3.pte
│
├── [HotSwapModel 내부]
│   (a) 현재 모델의 IO 버퍼 보존 (캐시 데이터 유지!)
│   (b) 새 모델(.pte) 로드 → NPU에 바인딩
│   (c) 보존된 캐시 버퍼를 새 모델에 재연결
│   (d) IO 크기 업데이트 (128→1 배치)
│
└── mTokenBatchSize = 1
```

**왜 모델 스왑이 필요한가?**

MediaTek NPU는 고정 shape 모델만 지원합니다:
- Prompt 모델: 입력 `[128, 4096]` — 한 번에 128 토큰 처리 (prefill 효율)
- Gen 모델: 입력 `[1, 4096]` — 한 번에 1 토큰 처리 (auto-regressive)

KV 캐시는 두 모델 간에 공유되므로, 스왑 시 캐시 데이터를 보존합니다.

---

## 10. Generation 단계 — gen_response()

**파일**: `mtk_llama_runner.cpp:197`

```
gen_response(llama_runtime, tokenizer, input_token=578)
│
├── [1] 모델 스왑: Prompt → Gen
│   llama_runtime.SwapModel(1)
│   → 128 배치 모델 → 1 배치 모델로 교체 (캐시 유지)
│
├── [2] 첫 토큰 디코드
│   decode(578, 578) → "The"
│   full_response = "The"
│   token_callback("The") → stdout에 "The" 출력
│
├── [3] 자기회귀 생성 루프
│   while (gen_tok_count < 50 && tokenIndex < 8192):
│
│   ─── 반복 1 (gen_tok_count=0) ───
│   │ logits = llama_runtime.Run({578})
│   │   → Run 내부:
│   │     padSize = 1 - 1 = 0 (패딩 없음!)
│   │     tokenEmbLut.lookupEmbedding({578}) → [1, 4096]
│   │     4개 청크 순차 실행 (각 1 토큰)
│   │     → logits [1, 128000]
│   │   mTokenIndex: 8 → 9
│   │
│   │ output_token = argmax<float>(logits, 128000) → 7438
│   │ decode(578, 7438) → " meaning"
│   │ token_callback(" meaning") → stdout 출력
│   │
│   ─── 반복 2 (gen_tok_count=1) ───
│   │ logits = llama_runtime.Run({7438})
│   │ output_token = argmax → 315
│   │ decode → " of" → stdout
│   │
│   ─── ... (계속) ───
│   │
│   ─── 반복 N: EOS 생성 ───
│   │ output_token = 128001 (EOS)
│   │ output_token == tokenizer->eos_tok() → true!
│   │ token_callback("</eos>")
│   │ break
│
├── [4] 생성 결과 출력
│   "[Generated Tokens]"
│   "{578, 7438, 315, 2324, ..., 128001}"
│
└── [5] 성능 통계
    "Token generation speed: 15.2 tok/s"
    (gen_tok_count / gen_total_time_sec)
```

### Meta vs MediaTek 샘플링 비교

```
Meta TextLLMRunner:
  Sampler(vocab_size=32000, temperature=0.8)
  → softmax(logits * 1.25)
  → top-p(0.9) 샘플링
  → 확률적 출력 (매번 다름)

MediaTek MTKLlamaRunner:
  argmax<float>(logits, 128000)
  → 단순 최대값 인덱스
  → 항상 greedy (결정적)
  → temperature, top-p 미지원
```

---

## 11. 헬퍼 라이브러리 상세 분석

### 11.1 TokenEmbeddingLut — 토큰 임베딩 룩업

```
구조: embedding.bin (mmap으로 로드)
  offset 0:        token 0의 임베딩 [float × 4096]
  offset 16384:    token 1의 임베딩 [float × 4096]
  ...
  offset 128000×16384: token 127999의 임베딩

lookupEmbedding([0, 0, ..., 128000, 3923, ...]):
  각 토큰 ID에 대해:
    src = mmap_base + tokenId × 4096 × 4
    dst = output_buffer + i × 4096 × 4
    memcpy(dst, src, 4096 × 4)  ← 단순 메모리 복사
```

### 11.2 RotaryEmbeddingMasterLut — RoPE 테이블

```
초기화:
  max_token_length = 8192
  rotEmbDim = 128 (= head_dim)
  rot_emb_base = 500000

generate():
  for pos in 0..8191:
    for dim in 0..63:
      freq = 1.0 / (500000 ^ (2*dim/128))
      cos_table[pos][dim] = cos(pos * freq)
      sin_table[pos][dim] = sin(pos * freq)

setEmbedding(tokenIndex=0, numTokens=8, padSize=120):
  Left-padding인 경우:
    output[0:120] = cos/sin at position 0 (패딩 위치)
    output[120] = cos/sin at position 0 (실제 token 0)
    output[121] = cos/sin at position 1
    ...
    output[127] = cos/sin at position 7
```

### 11.3 MaskBuilder — 어텐션 마스크

```
Prefill 시 마스크 (128×640):
  cache_size=512, batch_size=128
  mask 크기 = [128, 512+128] = [128, 640]

  Left-padding (padSize=120):
    행 0~119 (패딩): 전체 -inf (아무것도 attend 안 함)
    행 120 (실제 token 0): cache[0]만 attend
    행 121 (실제 token 1): cache[0:1] attend
    ...
    행 127 (실제 token 7): cache[0:7] attend (causal mask)

Gen 시 마스크 (1×513):
  cache_size=512, batch_size=1
  mask 크기 = [1, 512+1] = [1, 513]

  행 0 (현재 token): cache[0:8] attend + 자기 자신
```

---

## 12. Model-Specific vs 공용 부분 구분

### 공용 (Model-Agnostic) 레이어

| 컴포넌트 | 파일 | 재사용 가능 이유 |
|----------|------|-----------------|
| `IRunner` 인터페이스 | `extension/llm/runner/irunner.h` | 순수 인터페이스 |
| `GenerationConfig` | `extension/llm/runner/irunner.h` | 범용 생성 설정 |
| `ModelChunk` | `llama_runner/ModelChunk.h` | 모델 독립적 청크 관리 |
| `MultiModelLoader` | `llama_runner/MultiModelLoader.h` | 템플릿 기반 범용 로더 |
| `FileMemMapper` | `llama_runner/FileMemMapper.h` | 범용 파일 mmap |
| `LLMType` 시스템 | `llm_helper/llm_types.h` | 범용 타입 정의 |
| `Timer` | `llama_runner/Utils.h` | 범용 타이머 |
| `argmax` | `llama_runner/Utils.h` | 범용 argmax |
| `TokenEmbeddingLut` | `llm_helper/token_embedding.h` | 모델 독립적 LUT |
| `RotaryEmbeddingMasterLut` | `llm_helper/rotary_embedding.h` | RoPE 파라미터만 다름 |
| `MaskBuilder` | `llm_helper/mask_builder.h` | 캐시/배치 크기만 다름 |
| `Tokenizer` (pytorch/tokenizers) | 외부 라이브러리 | 완전 독립 |

### Model-Specific 레이어

| 컴포넌트 | 파일 | 모델 종속 이유 |
|----------|------|---------------|
| `LlamaModelOptions` | `LlamaConfig.h` | LLaMA 전용 모델 구조 파라미터 |
| `LlamaModelPaths` | `LlamaConfig.h` | LLaMA 전용 파일 경로 구조 |
| `LlamaModelChunk` | `LlamaModelChunk.h` | LLaMA IO 레이아웃 (Emb, Mask, RotEmb, Cache, Logits) |
| `LlamaRuntime` | `LlamaRuntime.h` | LLaMA 추론 파이프라인 (embedding → chunks → logits) |
| `MTKLlamaRunner` | `mtk_llama_runner.h` | LLaMA IRunner 구현 |
| `llama_runner_values.h` | `llm_helper/include/` | 하드코딩된 LLaMA 3 상수 |
| `mtk_llama_executor_runner.cpp` | main() | LLaMA 전용 CLI |

### 경계선 (부분적으로 재사용 가능)

| 컴포넌트 | 재사용 가능 부분 | 모델 종속 부분 |
|----------|-----------------|---------------|
| `LlamaModelChunk` | IO 관리, 패딩 처리, 캐시 롤백 | `defineIOs()`의 IO 순서/종류 |
| `LlamaRuntime` | 청크 오케스트레이션, 모델 스왑 | 초기화 시 컴포넌트 조합 |

---

## 13. 신규 모델 추가 가이드

### 예시: MediaTek NPU에서 Phi-3-Mini를 실행한다고 가정

#### 단계 1: 모델 설정 정의

```cpp
// PhiConfig.h — 새로 생성
struct PhiModelOptions {
  size_t prompt_token_batch_size = 64;  // Phi-3는 더 작은 배치
  size_t cache_size = 1024;
  size_t hidden_size = 3072;             // Phi-3 Mini: 3072
  size_t num_head = 32;
  size_t num_layer = 32;
  size_t head_dim = 96;                  // 3072/32 = 96
  size_t window_size = 0;
  size_t max_token_length = 4096;
  double partial_rotary_factor = 0.4;    // Phi-3 특유: 부분 회전 (40%)
  double rot_emb_base = 10000.0;

  LLMType model_input_type = LLMType::FP16;  // Phi-3는 FP16
  LLMType model_output_type = LLMType::FP16;
  LLMType cache_type = LLMType::FP16;
  LLMType mask_type = LLMType::FP16;
  LLMType rot_emb_type = LLMType::FP16;
};

struct PhiModelPaths {
  std::string tokenizer_path;
  std::string token_embedding_path;
  std::vector<std::string> prompt_model_paths;
  std::vector<std::string> gen_model_paths;
};
```

#### 단계 2: 모델 전용 ModelChunk 구현

```cpp
// PhiModelChunk.h — LlamaModelChunk를 참고하여 새로 생성
class PhiModelChunk : public ModelChunk {
  // Phi-3의 IO 레이아웃이 LLaMA와 다를 경우:
  void defineIOs() override {
    // Phi-3는 partial rotary embedding을 사용하므로
    // rotary와 non-rotary 입력이 분리될 수 있음
    defineInput(IOKind::Embedding, 1);
    defineInput(IOKind::Mask, 1);
    defineInput(IOKind::RotEmb, 1);
    defineInput(IOKind::KVCache, numCache);  // K+V 캐시
    defineOutput(IOKind::Embedding, 1);       // 또는 Logits
    defineOutput(IOKind::KVCache, numCache);
  }
};
```

#### 단계 3: Runtime 구현

```cpp
// PhiRuntime.h — LlamaRuntime을 참고
class PhiRuntime {
  void Initialize(const PhiModelOptions&, const PhiModelPaths&) {
    // (1) RotaryEmbedding: partial_rotary_factor=0.4 주의
    //     rotEmbDim = 96 * 0.4 = 38 (38차원만 회전)
    // (2) TokenEmbeddingLut: hidden_size=3072
    // (3) 모델 청크 생성/연결: 동일 패턴
  }
};
```

#### 단계 4: IRunner 구현

```cpp
// MTKPhiRunner.h
class MTKPhiRunner : public IRunner {
  // generate(), inference(), digest_prompt(), gen_response() 구현
  // 대부분 MTKLlamaRunner와 동일한 패턴
  // 차이점: 토크나이저 타입, 모델 옵션, BOS/EOS 토큰 ID
};
```

#### 단계 5: main() 작성

```cpp
// mtk_phi_executor_runner.cpp
// gflags로 Phi-3 전용 파라미터 정의
// 기존 mtk_llama_executor_runner.cpp와 거의 동일한 구조
```

### 변경 필요 사항 요약

| 레이어 | 변경 필요 | 재사용 가능 |
|--------|----------|------------|
| **ModelChunk 기반 클래스** | - | ✅ 그대로 사용 |
| **MultiModelLoader** | - | ✅ 그대로 사용 |
| **TokenEmbeddingLut** | - | ✅ hidden_size만 다름 |
| **RotaryEmbeddingMasterLut** | partial_rotary_factor 확인 | ✅ 파라미터만 변경 |
| **MaskBuilder** | SWA window_size 확인 | ✅ 파라미터만 변경 |
| **ModelConfig 구조체** | 새로 정의 | ❌ 모델마다 다름 |
| **ModelChunk IO 정의** | defineIOs() 재정의 | ⚠️ IO 레이아웃이 같으면 재사용 |
| **Runtime** | 초기화 로직 수정 | ⚠️ 구조 유사하지만 모델마다 다를 수 있음 |
| **IRunner 구현** | 새로 구현 | ⚠️ 패턴 동일, 세부 사항 다름 |
| **main()** | 새로 작성 | ⚠️ 플래그만 다름 |
| **argmax/Timer/Utils** | - | ✅ 그대로 사용 |

### 신규 모델 추가 시 예상 작업량

| 항목 | 코드량 (LOC) | 난이도 |
|------|-------------|-------|
| Config 구조체 | ~50 | 낮음 |
| ModelChunk (IO 동일 시) | 0 (재사용) | - |
| ModelChunk (IO 다를 시) | ~200 | 중간 |
| Runtime | ~150 | 중간 |
| IRunner 구현 | ~200 | 낮음 (패턴 복사) |
| main() | ~100 | 낮음 |
| 모델 export 스크립트 | ~300 | 높음 (Python) |
| 임베딩 파일 추출 | ~50 | 낮음 |
| **합계** | **~750~1050** | - |

---

## 14. Meta TextLLMRunner와의 상세 비교

| 측면 | Meta TextLLMRunner | MediaTek MTKLlamaRunner |
|------|-------------------|------------------------|
| **실행 장치** | CPU (XNNPACK) | MediaTek NPU (Neuron) |
| **모델 파일** | 1개 .pte | 8개 .pte (4 prompt + 4 gen) |
| **입력 shape** | 동적 [1, N] | 고정 [128, _] 또는 [1, _] |
| **KV 캐시** | 모델 내부 자동 관리 | 외부 수동 관리 (버퍼 할당, 롤백) |
| **Token Embedding** | 모델 내장 | 외부 파일 (embedding.bin, mmap) |
| **Rotary Embedding** | 모델 내장 | 외부 계산 (마스터 LUT → 슬라이스) |
| **Attention Mask** | 모델 내장 (SDPA) | 외부 빌드 (MaskBuilder) |
| **모델 스왑** | 불필요 | prompt↔gen 핫스왑 (멀티스레드) |
| **패딩** | 불필요 (동적 shape) | 필수 (left/right + 캐시 롤백) |
| **샘플링** | temperature + top-p | argmax only (greedy) |
| **IOManager** | Module이 자동 관리 | 수동 버퍼 링크 (input↔output) |
| **메모리 할당** | 표준 malloc | NeuronBufferAllocator (NPU) |
| **스레드풀** | XNNPACK 내부 | NPU 하드웨어 병렬성 |
| **데이터 타입** | Float/Half/BFloat16 | INT4/INT8/INT16/FP16/FP32 |
| **통계** | Stats 구조체 + JSON | Timer 콜백 (간단) |
| **SWA** | 미지원 | 지원 (window_size > 0) |

### 왜 이런 차이가 존재하는가?

**NPU의 제약:**
1. **고정 shape**: NPU 컴파일러가 텐서 크기를 컴파일 타임에 결정 → 동적 shape 불가
2. **메모리 제한**: NPU 전용 메모리가 제한적 → 모델 청크 분할 필요
3. **연산 제한**: NPU가 지원하지 않는 연산(embedding lookup, RoPE, mask)은 CPU에서 처리
4. **버퍼 관리**: NPU 버퍼는 특수 할당자 필요 → NeuronBufferAllocator

**결과적으로 MediaTek은:**
- 모델을 "순수 트랜스포머 블록"으로 분리하여 NPU에서 실행
- embedding, RoPE, mask, KV 캐시는 CPU에서 외부 관리
- prompt/gen 별도 컴파일로 배치 크기 문제 해결

---

## 15. 전체 호출 스택 요약

```
main()
│
├── runtime_init()
├── gflags::ParseCommandLineFlags()
├── get_model_options() → LlamaModelOptions
├── get_model_paths() → LlamaModelPaths
│
├── load_tokenizer() → TikToken/BPE/HF
│
├── LlamaRuntime::Initialize(options, paths)
│   ├── RotaryEmbeddingMasterLut(FP32, 8192, 128, 500000)
│   │   └── generate() → cos/sin 테이블 [8192, 128]
│   ├── for i in 0..3:
│   │     LlamaModelChunk(pathMap, options, batch=128, numCache=16)
│   ├── chunk[1..3].SetInputBuffer(prev.GetOutputBuffer())
│   ├── chunk[0..3].Initialize()
│   │   └── NPU 모델 로드, 버퍼 할당, IO 바인딩
│   ├── TokenEmbeddingLut("embedding.bin", FP32, 4096)
│   │   └── mmap → vocab_size=128000
│   └── tokenEmbLut.setOutput(chunk[0].inputBuffer)
│
├── inference(runtime, tokenizer, prompt)
│   ├── tokenizer->encode(prompt, bos=1, eos=0)
│   │   → [128000, 3923, 374, 279, 7438, 315, 2324, 30]
│   │
│   ├── digest_prompt(runtime, tokenizer, tokens)
│   │   └── while (more tokens):
│   │       getNextTokens() → 8 tokens (+ 120 padding)
│   │       runtime.Run([0,...,0, 128000, 3923, ..., 30])
│   │       ├── Left-padding: 120 + 8 = 128 tokens
│   │       ├── tokenEmbLut.lookupEmbedding() → [128, 4096]
│   │       ├── for chunk in chunks:
│   │       │     chunk.SetLeftPadding(120)
│   │       │     chunk.Run()
│   │       │     ├── UpdatePosEmbAndMask()
│   │       │     │   ├── RotaryEmbedding.setEmbedding()
│   │       │     │   └── MaskBuilder.buildMask()
│   │       │     ├── PrepareCacheIOs()
│   │       │     ├── SetBackendInputs/Outputs()
│   │       │     ├── GetModelMethod().execute() ← ★ NPU 실행
│   │       │     ├── PaddingPostprocess()
│   │       │     │   └── LeftPaddingCachePostprocess()
│   │       │     │       └── RollbackCache(120, 8)
│   │       │     └── AdvanceTokenIndex() → 0→8
│   │       ├── mTokenIndex += 8
│   │       └── return logits → argmax → 578
│   │
│   └── gen_response(runtime, tokenizer, 578)
│       ├── runtime.SwapModel(1) ← 128배치→1배치
│       │   └── 4 threads: chunk[i].HotSwapModel(1)
│       ├── decode(578, 578) → "The"
│       ├── token_callback("The")
│       └── while (count < 50 && tokenIndex < 8192):
│           ├── runtime.Run({output_token})
│           │   ├── padSize = 0 (1배치, 1토큰)
│           │   ├── tokenEmbLut.lookupEmbedding() → [1, 4096]
│           │   ├── 4 chunks 실행 (1토큰씩)
│           │   └── return logits → argmax → next_token
│           ├── EOS 체크: output_token == 128001? → break
│           ├── decode(prev, cur) → text
│           └── token_callback(text) → stdout
│
└── LlamaRuntime::Release()
    └── chunks clear, rotEmb reset, tokenEmb reset
```
