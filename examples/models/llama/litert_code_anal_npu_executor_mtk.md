# Gemma3-1B-IT NPU Executor 전체 코드 플로우 상세 분석

> **대상 파일**: `Gemma3-1B-IT_q4_ekv1280_mt6989.litertlm` (984 MB)
> **진입점**: `runtime/engine/litert_lm_main.cc`
> **핵심 Executor**: `runtime/executor/llm_litert_npu_compiled_model_executor.cc`
> **플랫폼**: MediaTek Dimensity 9300 (mt6989) APU / NeuroPilot
> **모델 사양**: Gemma3-1B-IT, INT4 weight quantization, max KV length 1280

---

## 목차

1. [실행 예시 시나리오 설정](#scenario)
2. [Phase 1: 프로그램 시작 및 모델 로딩](#phase1)
3. [Phase 2: Engine 및 Session 생성](#phase2)
4. [Phase 3: NPU Executor 초기화 — 텐서 버퍼 할당](#phase3)
5. [Phase 4: Conversation 및 Tokenization](#phase4)
6. [Phase 5: Prefill — 128 토큰 청크 처리](#phase5)
7. [Phase 6: Decode — Autoregressive 토큰 생성](#phase6)
8. [Phase 7: 응답 완성 및 벤치마크](#phase7)
9. [전체 텐서 데이터 흐름 요약도](#summary)
10. [부록: Gemma3-1B 모델 아키텍처 상수](#appendix)

---

<a name="scenario"></a>
## 1. 실행 예시 시나리오 설정

이 문서 전체에서 다음의 **구체적인 실행 예시**를 사용한다:

```bash
./litert_lm_main \
  --backend=npu \
  --model_path=Gemma3-1B-IT_q4_ekv1280_mt6989.litertlm \
  --input_prompt="What is the tallest building in the world?"
```

### Gemma3 Chat Template 적용 후 실제 프롬프트

Gemma3-IT 모델은 Jinja 기반 chat template을 사용한다. 위 사용자 프롬프트에 template이 적용되면:

```
<start_of_turn>user
What is the tallest building in the world?<end_of_turn>
<start_of_turn>model
```

### SentencePiece Tokenization 결과 (예시)

| 위치 | 토큰 텍스트 | Token ID |
|---:|:---|---:|
| 0 | `<bos>` | 2 |
| 1 | `<start_of_turn>` | 106 |
| 2 | `user` | 1645 |
| 3 | `\n` | 108 |
| 4 | `What` | 1841 |
| 5 | `is` | 603 |
| 6 | `the` | 573 |
| 7 | `tallest` | 97301 |
| 8 | `building` | 4621 |
| 9 | `in` | 575 |
| 10 | `the` | 573 |
| 11 | `world` | 2134 |
| 12 | `?` | 235336 |
| 13 | `<end_of_turn>` | 107 |
| 14 | `\n` | 108 |
| 15 | `<start_of_turn>` | 106 |
| 16 | `model` | 2516 |
| 17 | `\n` | 108 |

**총 토큰 수: 18개** (실제 값은 SentencePiece vocab에 따라 다를 수 있으나, 이 문서에서는 18개로 가정)

---

<a name="phase1"></a>
## 2. Phase 1: 프로그램 시작 및 모델 로딩

### 2.1 `main()` → `MainHelper()`

```
main(argc, argv)
  └── MainHelper(argc, argv)
        ├── absl::ParseCommandLine(argc, argv)
        ├── model_path = "Gemma3-1B-IT_q4_ekv1280_mt6989.litertlm"
        ├── backend_str = "npu"
        └── ModelAssets::Create(model_path)
```

### 2.2 ModelAssets::Create() — .litertlm 파일 열기

`ModelAssets`는 모델 파일 경로를 가지고 `ScopedFile`(파일 디스크립터) 또는 `MemoryMappedFile`(mmap)을 생성한다.

```cpp
// executor_settings_base.h
class ModelAssets {
  std::string path_;                          // 파일 경로
  std::shared_ptr<ScopedFile> scoped_file_;   // fd 래퍼
  std::shared_ptr<MemoryMappedFile> memory_mapped_file_;  // mmap 버퍼
};
```

### 2.3 .litertlm 파일 물리적 레이아웃

984 MB 파일의 내부 구조:

```
┌─────────────────────────────────────────────────────────────┐
│ [0x00..0x07]  Magic: "LITERTLM"                              │
│ [0x08..0x0B]  version_major: 1                               │
│ [0x0C..0x0F]  version_minor: 1         ← v1.1 (NPU 모델)    │
│ [0x10..0x13]  version_patch: 0                               │
│ [0x14..0x17]  padding: 0x00000000                            │
│ [0x18..0x1F]  header_end_offset: 4,734,976                   │
│ [0x20..0x24F] FlatBuffer: LiteRTLMMetaData (560 bytes)       │
│               ... alignment padding ...                       │
├─────────────────────────────────────────────────────────────┤
│ Section 0: LlmMetadataProto (117 bytes)                      │
│   offset: 16,384 → 16,501                                    │
│   내용: start_token, stop_tokens, sampler_params, max_tokens  │
├─────────────────────────────────────────────────────────────┤
│ Section 1: SP_Tokenizer (~4.5 MB)                            │
│   offset: 32,768 → 4,721,840                                 │
│   내용: SentencePiece 모델 바이너리 (vocab_size=262,144)      │
├─────────────────────────────────────────────────────────────┤
│ Section 2: TF_LITE_EMBEDDER (150 MB)                         │
│   offset: 4,734,976 → 162,025,160                            │
│   subgraphs: decode_embedder, prefill_embedder_128            │
│   핵심 weight: INT4 [262144, 1152] (embedding table)          │
├─────────────────────────────────────────────────────────────┤
│ Section 3: TF_LITE_AUX (~116 KB)                             │
│   offset: 162,037,760 → 162,154,352                          │
│   subgraphs: decode/prefill × {cache_update, mask, rope}     │
├─────────────────────────────────────────────────────────────┤
│ Section 4: TF_LITE_PREFILL_DECODE (830 MB)                   │
│   offset: 162,168,832 → 1,032,825,728                        │
│   subgraphs: decode (1개 APU op), prefill_128 (1개 APU op)   │
│   ← 전체 Transformer 26-layer가 단일 APU delegate에 컴파일   │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Backend 결정

```cpp
// litert_lm_main.cc:131-132
ASSIGN_OR_RETURN(Backend backend,
                 litert::lm::GetBackendFromString("npu"));
// → Backend::NPU
```

### 2.5 EngineSettings::CreateDefault()

```cpp
ASSIGN_OR_RETURN(
    EngineSettings engine_settings,
    EngineSettings::CreateDefault(std::move(model_assets), Backend::NPU));
```

내부적으로 `LlmExecutorSettings`를 생성하며, 모델 메타데이터에서 `max_num_tokens=1280`을 읽어온다.

---

<a name="phase2"></a>
## 3. Phase 2: Engine 및 Session 생성

### 3.1 EngineFactory::CreateAny()

```
EngineFactory::CreateAny(engine_settings)
  └── EngineFactory::Create(kLiteRTCompiledModel, settings)
        └── EngineImpl::Create(settings)    // runtime/core/engine_impl.cc
```

### 3.2 EngineImpl::Create() 내부 흐름

```
EngineImpl::Create(EngineSettings)
  ├── 1. LiteRT Environment 생성 (싱글톤)
  │     └── litert::Environment::Create()
  │
  ├── 2. ModelResources 로딩 (lazy-load)
  │     └── .litertlm → 섹션별 TFLite 모델 추출
  │         ├── kTfLitePrefillDecode  → 830 MB (APU 바이너리)
  │         ├── kTfLiteEmbedder       → 150 MB (CPU embedding)
  │         └── kTfLiteAux            → 116 KB (CPU mask/rope/cache)
  │
  ├── 3. Tokenizer 초기화
  │     └── SentencePiece 로딩 (4.5 MB)
  │         vocab_size = 262,144
  │
  ├── 4. LlmMetadata 파싱
  │     ├── start_token_id = 2 (<bos>)
  │     ├── stop_token_ids = [[1], [107]]  (<eos>, <end_of_turn>)
  │     ├── max_num_tokens = 1280
  │     └── prompt_templates / jinja_template
  │
  ├── 5. Executor 생성 (Backend::NPU)
  │     └── LlmLiteRtNpuCompiledModelExecutor::Create()
  │         (아래 Phase 3에서 상세 설명)
  │
  └── 6. EngineImpl 인스턴스 반환
```

### 3.3 Session 및 Conversation 생성

```cpp
// litert_lm_main.cc:145-152
auto session_config = SessionConfig::CreateDefault();
// SessionConfig 기본값:
//   sampler_params: temperature=1.0, top_k=1 (greedy)
//   start_token_id = 2
//   stop_token_ids = [[1], [107]]
//   max_output_tokens = INT_MAX

ASSIGN_OR_RETURN(auto conversation_config,
                 ConversationConfig::Builder()
                     .SetSessionConfig(session_config)
                     .Build(*engine));

ASSIGN_OR_RETURN(conversation,
                 Conversation::Create(*engine, conversation_config));
```

`Conversation::Create()`는 내부적으로:
1. `engine->CreateSession(session_config)` 호출
2. `ModelDataProcessor` 생성 (Gemma3DataProcessor — tokenization + chat template)
3. Preface가 있으면 prefill 실행 (이 예에서는 없음)

---

<a name="phase3"></a>
## 4. Phase 3: NPU Executor 초기화 — 텐서 버퍼 할당

### 4.1 `LlmLiteRtNpuCompiledModelExecutor::Create()`

```
Create(executor_settings, resources, env)
  ├── resources.GetTFLiteModel(kTfLitePrefillDecode) → llm_model (830 MB)
  ├── HasPerLayerEmbedder(*llm_model) → false  (Gemma3-1B는 per-layer 없음)
  └── CreateForModelWithoutPerLayerEmbedding(...)
```

Gemma3-1B-IT는 per-layer embedding이 없으므로 (Gemma3n과 구별) `CreateForModelWithoutPerLayerEmbedding()` 경로를 탄다.

### 4.2 LiteRT Options 생성 (NPU 가속기 설정)

```cpp
litert::Expected<litert::Options> CreateLiteRtOptions() {
  auto options = ::litert::Options::Create();
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);  // 폴백

  // Qualcomm QNN 옵션 (현재 하드코딩)
  auto qnn_opts = ::litert::qualcomm::QualcommOptions::Create();
  qnn_opts.SetLogLevel(kOff);
  qnn_opts.SetHtpPerformanceMode(kBurst);  // 최대 성능
  options.AddOpaqueOptions(std::move(qnn_opts));
  return options;
}
```

> **참고**: mt6989 파일용이지만 현재 코드는 Qualcomm 옵션만 설정됨.
> MediaTek NeuroPilot 지원은 `MediatekOptions`로 추가 예정.

### 4.3 CompiledModel 생성

```cpp
CompiledModel llm_compiled_model =
    CompiledModel::Create(env, transformer_model->Get(), options);
```

이 시점에서 LiteRT 런타임이:
1. TFLite 모델의 delegate op를 감지
2. QNN/NeuroPilot 플러그인 shared library 로드
3. 830 MB APU 바이너리를 NPU에 로딩

### 4.4 Transformer 버퍼 할당 (`AllocateTransformerBuffers`)

`prefill_128`과 `decode` 시그니처의 모든 입출력 텐서를 할당한다:

```
AllocateTransformerBuffers(env, transformer_model, llm_compiled_model, ...)
│
├── prefill_128 시그니처 입력 (59개):
│   ├── embeddings:      INT16 [1, 128, 1152]    ← Embedder 출력
│   ├── pos_emb_cos:     INT16 [1, 128, 1, 256]  ← RoPE 출력
│   ├── pos_emb_sin:     INT16 [1, 128, 1, 256]
│   ├── pos_emb_local_cos: INT16 [1, 128, 1, 256]
│   ├── pos_emb_local_sin: INT16 [1, 128, 1, 256]
│   ├── mask_global:     INT16 [1, 1, 128, 1408]  ← Mask 출력 (128+1280)
│   ├── mask_local:      INT16 [1, 1, 128, 1408]
│   ├── kv_cache_k_0..25: INT16 [1, 1, 1280, 256] × 26  ← KV cache 입력
│   └── kv_cache_v_0..25: INT16 [1, 1, 256, 1280] × 26
│
├── prefill_128 시그니처 출력 (52개):
│   ├── kv_slice_k_0..25: INT16 [1, 1, 128, 256] × 26   ← 새 K 슬라이스
│   └── kv_slice_v_0..25: INT16 [1, 1, 256, 128] × 26   ← 새 V 슬라이스
│   (⚠️ prefill에는 logits 출력 없음)
│
├── decode 시그니처 입력 (59개):
│   ├── embeddings:      INT16 [1, 1, 1152]       ← 단일 토큰
│   ├── pos_emb_cos:     INT16 [1, 1, 1, 256]
│   ├── pos_emb_sin:     INT16 [1, 1, 1, 256]
│   ├── pos_emb_local_cos: INT16 [1, 1, 1, 256]
│   ├── pos_emb_local_sin: INT16 [1, 1, 1, 256]
│   ├── mask_global:     INT16 [1, 1, 1, 1281]    ← (1 + 1280)
│   ├── mask_local:      INT16 [1, 1, 1, 1281]
│   ├── kv_cache_k_0..25: INT16 [1, 1, 1280, 256] × 26  ← 공유됨
│   └── kv_cache_v_0..25: INT16 [1, 1, 256, 1280] × 26
│
└── decode 시그니처 출력 (53개):
    ├── kv_slice_k_0..25: INT16 [1, 1, 1, 256] × 26
    ├── kv_slice_v_0..25: INT16 [1, 1, 256, 1] × 26
    └── logits:          INT16 [1, 1, 262144]      ← 다음 토큰 확률
```

### 4.5 버퍼 공유(Buffer Sharing) 아키텍처

핵심 설계: **zero-copy TensorBuffer 공유**로 sub-model 간 데이터 전달

```
                    ┌─────────────┐
                    │  Embedder   │ (CPU, 150MB)
                    │  CompiledModel │
 token_ids ────────►│  INT32 [1,128] │
                    │             │
                    └──────┬──────┘
                           │ Duplicate() (zero-copy)
                           ▼
            ┌──────────────────────────────┐
            │ LLM Transformer (NPU, 830MB) │
            │  embeddings: INT16 [1,128,1152] │◄── 공유 버퍼
            │                              │
            │  pos_emb_*   ◄───────────────│───── RoPE 출력 (공유)
            │  mask_*      ◄───────────────│───── Mask 출력 (공유)
            │  kv_cache_*  ◄───────────────│───── Cache Update 출력 (공유)
            │                              │
            │  → kv_slice_* ───────────────│────► Cache Update 입력 (공유)
            │  → logits                    │
            └──────────────────────────────┘
```

**`TensorBuffer::Duplicate()`**는 같은 underlying 메모리를 참조하는 새 핸들을 생성한다. 따라서 Embedder가 출력을 쓰면 LLM 입력에 즉시 반영된다.

### 4.6 개별 Context 생성

```
CreateForModelWithoutPerLayerEmbedding(...)
  ├── AllocateTransformerBuffers(...)         ← 4.4
  ├── CreateLlmInferenceContextWithBufferSharing(...)
  │     ← prefill/decode 양쪽의 input/output 버퍼를 InferenceContext로 묶음
  │     ← KV cache 버퍼는 prefill/decode간 공유
  │     ← decode만 logits 출력 버퍼를 가짐
  │
  ├── CreateNpuAuxiliaryContext(env, aux_model)  ← AUX CompiledModel (CPU)
  ├── CreateMaskContextWithBufferSharing(...)     ← mask 입출력 버퍼
  ├── CreateEmbedderContextWithBufferSharing(...) ← embedder 입출력 버퍼
  ├── CreateRopeContextWithBufferSharing(...)     ← RoPE 입출력 버퍼
  ├── CreateCacheUpdateInferenceContextWithBufferSharing(...)
  │     ← input_pos 공유 (RoPE와 동일 버퍼)
  │     ← kv_cache + kv_slice → updated kv_cache
  │
  └── WarmupInference(...)  ← 모든 모델 1회 워밍업 실행 후 KV cache 초기화
```

### 4.7 Warmup Inference

첫 실제 추론 전에 모든 모델을 한 번 실행하여 NPU 런타임을 초기화한다:

```
WarmupInference(...)
  ├── Fill(embeddings, 1)              ← 0으로 나누기 방지 (DIV op)
  ├── llm.Run("prefill_128", ...)      ← NPU 워밍업
  ├── llm.Run("decode", ...)           ← NPU 워밍업
  ├── aux.Run("prefill_rope_128", ...) ← CPU 워밍업
  ├── aux.Run("decode_rope", ...)
  ├── aux.Run("prefill_mask_128", ...)
  ├── aux.Run("decode_mask", ...)
  ├── aux.Run("prefill_cache_update_128", ...)
  ├── aux.Run("decode_cache_update", ...)
  └── ClearKVCache(...)                ← 모든 kv_cache 버퍼를 0으로 초기화
```

### 4.8 SortedPrefillSignatureMap

```cpp
// mt6989 NPU 모델은 128 토큰 prefill만 지원
SortedPrefillSignatureMap prefill_runner_set;
prefill_runner_set[128] = "prefill_128";
// ← CPU/GPU 모델은 {32, 64, 128, 256, 512, 1024, 2560} 지원
```

---

<a name="phase4"></a>
## 5. Phase 4: Conversation 및 Tokenization

### 5.1 `SendMessageAsync()` 호출

```cpp
// litert_lm_main.cc:162-164
conversation->SendMessageAsync(
    json::object({{"role", "user"}, {"content", content_list}}),
    CreateMessageCallback());
```

### 5.2 Conversation 내부 처리

```
Conversation::SendMessageAsync(json_message, callback)
  │
  ├── 1. history_ 업데이트: user 메시지 추가
  │
  ├── 2. GetSingleTurnText(history_) — Chat Template 적용
  │     └── Jinja template 렌더링:
  │         "<start_of_turn>user\nWhat is the tallest building in the world?
  │          <end_of_turn>\n<start_of_turn>model\n"
  │
  ├── 3. model_data_processor_->ToInputDataVector(rendered_text)
  │     └── SentencePiece tokenize:
  │         [2, 106, 1645, 108, 1841, 603, 573, 97301,
  │          4621, 575, 573, 2134, 235336, 107, 108, 106, 2516, 108]
  │         → TensorBuffer token_ids: INT32 [1, 18]
  │
  ├── 4. session_->RunPrefillAsync(inputs, callback_chain)
  │     └── Phase 5에서 상세 설명
  │
  └── 5. (prefill 완료 후) session_->RunDecodeAsync(callback)
        └── Phase 6에서 상세 설명
```

### 5.3 Token ID 배열 상태

```
token_ids (INT32) [1, 18]:
┌───┬─────┬──────┬─────┬──────┬─────┬─────┬───────┬──────┬─────┬─────┬──────┬────────┬─────┬─────┬─────┬──────┬─────┐
│ 2 │ 106 │ 1645 │ 108 │ 1841 │ 603 │ 573 │ 97301 │ 4621 │ 575 │ 573 │ 2134 │ 235336 │ 107 │ 108 │ 106 │ 2516 │ 108 │
└───┴─────┴──────┴─────┴──────┴─────┴─────┴───────┴──────┴─────┴─────┴──────┴────────┴─────┴─────┴─────┴──────┴─────┘
 bos  sot  user    \n   What   is   the  tallest build  in   the  world     ?     eot    \n   sot  model   \n
```

---

<a name="phase5"></a>
## 6. Phase 5: Prefill — 128 토큰 청크 처리

### 6.1 `Prefill()` → Work Group 계산

```cpp
// llm_litert_npu_compiled_model_executor.cc:963-998
Prefill(inputs, params)
  ├── tensor_type.Layout().Dimensions() == [1, 18]  (batch=1, seq=18)
  ├── ids = span<int32_t>{2, 106, 1645, ..., 108}  (18 tokens)
  │
  ├── GetOptimizedPrefillWorkGroups(prefill_signature_map_, 18)
  │     prefill_signature_map_ = {128: "prefill_128"}
  │     input_length = 18
  │     max_seq_len = 128
  │     18 < 128 이므로 while 루프 건너뜀
  │     → work_groups = [("prefill_128", 18)]
  │     (128 슬롯 중 18개만 사용, 나머지 110개는 0 패딩)
  │
  └── PrefillInternal("prefill_128", ids[0..18])
```

### 6.2 `PrefillInternal()` — Step-by-Step 텐서 준비

> **핵심**: 18개 토큰 중 마지막 1개(token 17: `108`)는 **pending token**으로 보류.
> 실제 prefill에 들어가는 것은 17개 + 가능한 이전 pending token.

```
PrefillInternal("prefill_128", ids[0..18])
  │
  ├── ① Embedder 입력 버퍼 잠금 (Lock)
  │     embedder_input: INT32 [1, 128] ← memset(0)
  │     rope_input_pos: INT32 [128]    ← memset(0)
  │     mask_timestep:  INT32 []       ← memset(0)
  │
  ├── ② Pending token 확인
  │     processed_tokens_.GetNextUnprocessedToken()
  │     → (step=0, pending_token=[])  (첫 호출이므로 없음)
  │     input_idx = 0
  │
  ├── ③ timestep 설정
  │     mask_timestep = 0  (internal_start_step)
  │
  ├── ④ ids[0..16] 을 버퍼에 채우기 (마지막 1개는 보류)
  │     for i in 0..16:  (17개 토큰)
  │       embedder_input[input_idx] = ids[i]
  │       rope_input_pos[input_idx] = current_step_  (0, 1, 2, ... 16)
  │       current_step_++
  │
  │     processed_tokens_.AddProcessedTokens([2,106,...,106,2516])
  │     current_step_ = 17
```

### 6.3 Embedder 입력 버퍼 최종 상태

```
embedder_input (INT32) [1, 128]:
┌───┬─────┬──────┬─────┬──────┬─────┬─────┬───────┬──────┬─────┬─────┬──────┬────────┬─────┬─────┬─────┬──────┬───┬───┬ ... ┬───┐
│ 2 │ 106 │ 1645 │ 108 │ 1841 │ 603 │ 573 │ 97301 │ 4621 │ 575 │ 573 │ 2134 │ 235336 │ 107 │ 108 │ 106 │ 2516 │ 0 │ 0 │ ... │ 0 │
└───┴─────┴──────┴─────┴──────┴─────┴─────┴───────┴──────┴─────┴─────┴──────┴────────┴─────┴─────┴─────┴──────┴───┴───┴ ... ┴───┘
 [0]  [1]   [2]   [3]   [4]   [5]   [6]   [7]     [8]   [9]  [10]  [11]   [12]    [13] [14] [15]  [16]  ← 17개 유효, 111개 0-패딩

rope_input_pos (INT32) [128]:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬────┬────┬────┬────┬────┬────┬────┬───┬ ... ┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │ 16 │ 0 │ ... │ 0 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴────┴────┴────┴────┴────┴────┴────┴───┴ ... ┴───┘

mask_timestep (INT32) scalar: 0
```

### 6.4 Pending Token 저장

```cpp
// ids.back() = 108 (마지막 "\n" 토큰)
last_input_token = TokenData(id=108)
processed_tokens_.AddPendingInputToken({last_input_token})
current_step_ = 18  (17 processed + 1 pending)
```

> **왜 마지막 토큰을 보류하는가?**
>
> Prefill 단계에서는 logits를 출력하지 않는다 (NPU 모델 특성). 마지막 토큰은 다음 Decode
> 단계의 첫 번째 입력이 되어 logits를 생성하는 데 사용된다.

### 6.5 Sub-model 순차 실행 (Prefill 경로)

#### ⑤ Embedder 실행 (CPU)

```
embedder.Run("prefill_embedder_128",
             input:  {token_ids: INT32 [1, 128]},
             output: {embeddings: INT16 [1, 128, 1152]})
```

**실행 과정**:
- INT4 [262144, 1152] embedding table에서 각 token ID에 대응하는 row를 lookup
- INT4 → INT16 dequantization 후 출력

```
embeddings (INT16) [1, 128, 1152]:
┌──────────────────────────────────────────────────────┐
│ [0]:  emb(<bos>)         → 1152-dim INT16 벡터       │  ← token 2
│ [1]:  emb(<start_of_turn>) → 1152-dim INT16 벡터     │  ← token 106
│ [2]:  emb("user")        → 1152-dim INT16 벡터       │  ← token 1645
│ ...                                                    │
│ [16]: emb("model")       → 1152-dim INT16 벡터       │  ← token 2516
│ [17]: emb(pad=0)         → zero 벡터                  │  ← 0-패딩
│ ...                                                    │
│ [127]: emb(pad=0)        → zero 벡터                  │  ← 0-패딩
└──────────────────────────────────────────────────────┘
```

> 이 출력은 zero-copy로 LLM Transformer의 `embeddings` 입력과 공유됨.

#### ⑥ RoPE 실행 (CPU, AUX model)

```
aux.Run("prefill_rope_128",
        input:  {input_pos: INT32 [128]},     ← [0,1,2,...,16,0,...,0]
        output: {pos_emb_cos:       INT16 [1, 128, 1, 256],
                 pos_emb_sin:       INT16 [1, 128, 1, 256],
                 pos_emb_local_cos: INT16 [1, 128, 1, 256],
                 pos_emb_local_sin: INT16 [1, 128, 1, 256]})
```

**Gemma3 RoPE 특징**: global과 local 두 종류의 RoPE를 생성
- **global**: 전체 context에 적용되는 position encoding (전체 attention layer용)
- **local**: sliding window 범위 내에서만 적용되는 position encoding

```
pos_emb_cos (INT16) [1, 128, 1, 256]:
┌──────────────────────────────────────────────────────────┐
│ [0,0,0,:]: cos(0 × θ_0), cos(0 × θ_1), ..., cos(0 × θ_127)   │  pos=0
│ [0,1,0,:]: cos(1 × θ_0), cos(1 × θ_1), ..., cos(1 × θ_127)   │  pos=1
│ ...                                                              │
│ [0,16,0,:]: cos(16 × θ_0), ..., cos(16 × θ_127)                │  pos=16
│ [0,17,0,:]: cos(0 × θ_0), ..., cos(0 × θ_127)                  │  pos=0 (패딩)
│ ...                                                              │
│ [0,127,0,:]: cos(0 × θ_0), ...                                  │  pos=0 (패딩)
└──────────────────────────────────────────────────────────┘
```

> 출력은 zero-copy로 LLM 입력의 `pos_emb_*`와 공유됨.

#### ⑦ Mask 실행 (CPU, AUX model)

```
aux.Run("prefill_mask_128",
        input:  {time_step:    INT32 scalar = 0,
                 input_tokens: INT32 [1, 128]},     ← embedder와 동일 토큰
        output: {mask_global: INT16 [1, 1, 128, 1408],
                 mask_local:  INT16 [1, 1, 128, 1408]})
```

**마스크 차원 설명**: `1408 = 128 (현재 prefill 시퀀스) + 1280 (KV cache 최대 길이)`

```
mask_global (INT16) [1, 1, 128, 1408]:
  ┌─── 1280 (KV cache) ───┬── 128 (current) ──┐
  │ all masked (0)         │ causal mask       │   ← 행 [0]: 자기 자신만 attend
  │ all masked (0)         │ lower triangular  │   ← 행 [1]: 토큰 0,1 attend
  │ ...                    │ ...               │
  │ all masked (0)         │ row 16: 0..16     │   ← 행 [16]: 토큰 0~16 attend
  │ all masked (0)         │ row 17..127: 패딩  │
  └────────────────────────┴───────────────────┘

mask_local (INT16) [1, 1, 128, 1408]:
  ← sliding window(512) 내 토큰만 attend 허용
  ← 첫 prefill이므로 global과 동일 패턴 (모든 토큰이 window 내)
```

#### ⑧ LLM Transformer 실행 (NPU/APU)

```
llm.Run("prefill_128",
        input:  {embeddings:       INT16 [1, 128, 1152],     ← Embedder 출력
                 pos_emb_cos:      INT16 [1, 128, 1, 256],   ← RoPE 출력
                 pos_emb_sin:      INT16 [1, 128, 1, 256],
                 pos_emb_local_cos: INT16 [1, 128, 1, 256],
                 pos_emb_local_sin: INT16 [1, 128, 1, 256],
                 mask_global:      INT16 [1, 1, 128, 1408],  ← Mask 출력
                 mask_local:       INT16 [1, 1, 128, 1408],
                 kv_cache_k_0..25: INT16 [1, 1, 1280, 256] × 26,  ← 0 초기화
                 kv_cache_v_0..25: INT16 [1, 1, 256, 1280] × 26},

        output: {kv_slice_k_0..25: INT16 [1, 1, 128, 256] × 26,   ← 새 KV
                 kv_slice_v_0..25: INT16 [1, 1, 256, 128] × 26})
```

**NPU 내부에서 일어나는 일** (26-layer Transformer, 단일 APU delegate):

```
Layer 0 ~ Layer 25 (반복):
  ┌─────────────────────────────────────────────────────┐
  │  Input Norm (RMSNorm)                                │
  │    x: [1, 128, 1152] → x_norm: [1, 128, 1152]       │
  │                                                       │
  │  Self-Attention (GQA: 4 heads, 1 KV head)            │
  │    Q = x_norm × W_Q: [1, 128, 1152] → [1, 4, 128, 256]  │
  │    K = x_norm × W_K: [1, 128, 1152] → [1, 1, 128, 256]  │
  │    V = x_norm × W_V: [1, 128, 1152] → [1, 1, 128, 256]  │
  │                                                       │
  │    Q = apply_rope(Q, pos_emb_cos/sin)                 │
  │    K = apply_rope(K, pos_emb_cos/sin)                 │
  │                                                       │
  │    K_full = concat(kv_cache_k[layer], K)              │
  │    V_full = concat(kv_cache_v[layer], V)              │
  │                                                       │
  │    if (global_attention_layer):                        │
  │      attn = softmax(Q @ K_full.T / √256) × mask_global │
  │    else:                                               │
  │      attn = softmax(Q @ K_full.T / √256) × mask_local  │
  │                                                       │
  │    output = attn @ V_full                              │
  │    output = output × W_O                               │
  │                                                       │
  │  output_kv_slice_k[layer] = K   ← [1, 1, 128, 256]   │
  │  output_kv_slice_v[layer] = V   ← [1, 1, 256, 128]   │
  │                                                       │
  │  Post-Attention Norm (RMSNorm)                        │
  │                                                       │
  │  FFN (GeGLU variant)                                  │
  │    gate = x × W_gate: [1,128,1152] → [1,128,6912]    │
  │    up   = x × W_up:   [1,128,1152] → [1,128,6912]    │
  │    h = gelu(gate) ⊙ up                                │
  │    out = h × W_down: [1,128,6912] → [1,128,1152]     │
  │                                                       │
  │  Residual connection                                   │
  └─────────────────────────────────────────────────────┘
```

**KV Slice 출력 상태** (prefill 후):

```
kv_slice_k_0 (INT16) [1, 1, 128, 256]:
  ┌──────────────────────────────────────────┐
  │ [0,0,0,:]: K for token 0 (<bos>)          │  256-dim 벡터
  │ [0,0,1,:]: K for token 1 (<sot>)          │
  │ ...                                        │
  │ [0,0,16,:]: K for token 16 ("model")      │
  │ [0,0,17,:]: K for token 17 (pad)          │  ← 패딩된 위치
  │ ...                                        │
  │ [0,0,127,:]: K for token 127 (pad)        │
  └──────────────────────────────────────────┘
  × 26 layers (kv_slice_k_0 ~ kv_slice_k_25)
```

#### ⑨ Cache Update 실행 (CPU, AUX model)

```
aux.Run("prefill_cache_update_128",
        input:  {kv_cache_k_0..25:  INT16 [1, 1, 1280, 256] × 26,   ← 기존 cache (0)
                 kv_cache_v_0..25:  INT16 [1, 1, 256, 1280] × 26,
                 kv_slice_k_0..25:  INT16 [1, 1, 128, 256] × 26,    ← LLM 출력
                 kv_slice_v_0..25:  INT16 [1, 1, 256, 128] × 26,
                 input_pos:         INT32 [128]},                     ← [0,1,...,16,0,...,0]

        output: {kv_cache_k_0..25:  INT16 [1, 1, 1280, 256] × 26,   ← 업데이트됨
                 kv_cache_v_0..25:  INT16 [1, 1, 256, 1280] × 26})
```

**Cache Update 동작**:
- `input_pos`의 위치에 `kv_slice` 값을 삽입
- 위치 0~16에 실제 KV 값이, 나머지는 0

```
kv_cache_k_0 (INT16) [1, 1, 1280, 256] — After Prefill:
  ┌──────────────────────────────────────────────────┐
  │ [0,0,0,:]:    K_layer0(token 0)   ← 실제 데이터   │
  │ [0,0,1,:]:    K_layer0(token 1)                    │
  │ ...                                                │
  │ [0,0,16,:]:   K_layer0(token 16)  ← 마지막 유효    │
  │ [0,0,17,:]:   0 0 0 ... 0         ← 미사용         │
  │ ...                                                │
  │ [0,0,1279,:]: 0 0 0 ... 0         ← 미사용         │
  └──────────────────────────────────────────────────┘
```

### 6.6 Prefill 완료 후 상태 요약

| 항목 | 값 |
|:---|:---|
| `current_step_` | 18 |
| `processed_tokens_.TokenCount()` | 18 (17 processed + 1 pending) |
| pending token | `TokenData(id=108)` ("\n") |
| KV cache 유효 위치 | 0~16 (17개 토큰) |
| KV cache 남은 용량 | 1280 - 17 = 1263 슬롯 |
| Prefill work groups | 1회 (`prefill_128` × 18 tokens) |

---

<a name="phase6"></a>
## 7. Phase 6: Decode — Autoregressive 토큰 생성

### 7.1 `Decode()` — 첫 번째 Decode Step (Step 18)

```cpp
// llm_litert_npu_compiled_model_executor.cc:1001-1061
Decode(output_tokens, decode_params)
  │
  ├── decoded_logits = llm_inference_context_.decode_output_buffers["logits"]
  │     → INT16 [1, 1, 262144]
  │
  ├── processed_tokens_.GetNextUnprocessedToken()
  │     → (step=17, pending_token=[TokenData(id=108)])
  │     ← pending token이 있으므로 이것을 decode 입력으로 사용
  │
  ├── DecodeInternal(step=17, token=TokenData(108))
  │     → (아래 상세)
  │
  ├── processed_tokens_.MarkPendingInputTokenAsProcessed()
  │
  ├── ApplyGreedySampling(decoded_logits)
  │     → argmax(INT16 [262144]) → max_index (예: 651 = "The")
  │
  ├── last_output_token = TokenData(id=651)
  ├── processed_tokens_.AddPendingInputToken({last_output_token})
  ├── current_step_ = 19
  │
  └── output_tokens.Write({651})  ← 출력 토큰 기록
```

### 7.2 `DecodeInternal(step=17, token=108)` — Step-by-Step

#### ① 입력 텐서 준비

```
decode_input (INT32) [1, 1]:
┌─────┐
│ 108 │  ← pending token "\n"
└─────┘

rope_input_pos (INT32) [1]:
┌────┐
│ 17 │  ← current step
└────┘

mask_timestep (INT32) scalar:
┌────┐
│ 17 │
└────┘
```

#### ② Embedder 실행 (CPU)

```
embedder.Run("decode_embedder",
             input:  {token_ids:  INT32 [1, 1] = [108]},
             output: {embeddings: INT16 [1, 1, 1152]})
```

```
embeddings (INT16) [1, 1, 1152]:
┌──────────────────────────────────────────┐
│ emb("\n") → 1152-dim INT16 벡터           │
│ [e_0, e_1, e_2, ..., e_1151]             │
└──────────────────────────────────────────┘
```

#### ③ RoPE 실행 (CPU)

```
aux.Run("decode_rope",
        input:  {input_pos: INT32 [1] = [17]},
        output: {pos_emb_cos:       INT16 [1, 1, 1, 256],    ← cos(17 × θ)
                 pos_emb_sin:       INT16 [1, 1, 1, 256],    ← sin(17 × θ)
                 pos_emb_local_cos: INT16 [1, 1, 1, 256],
                 pos_emb_local_sin: INT16 [1, 1, 1, 256]})
```

```
pos_emb_cos (INT16) [1, 1, 1, 256]:
┌──────────────────────────────────────────────────────────┐
│ cos(17×θ_0), cos(17×θ_1), cos(17×θ_2), ..., cos(17×θ_127) │
│ (+ interleaved pattern for head_dim=256)                     │
└──────────────────────────────────────────────────────────┘
```

#### ④ Mask 실행 (CPU)

```
aux.Run("decode_mask",
        input:  {time_step:    INT32 scalar = 17,
                 input_tokens: INT32 [1, 1] = [108]},
        output: {mask_global: INT16 [1, 1, 1, 1281],
                 mask_local:  INT16 [1, 1, 1, 1281]})
```

**마스크 차원**: `1281 = 1 (현재 토큰) + 1280 (KV cache)`

```
mask_global (INT16) [1, 1, 1, 1281]:
┌─── 1280 (KV cache positions) ──────────────────────┬── 1 ──┐
│ pos[0..16]=attend, pos[17..1279]=masked             │ self  │
└─────────────────────────────────────────────────────┴───────┘

mask_local (INT16) [1, 1, 1, 1281]:  ← sliding window = 512
│ max(0, 17-512)..16 = pos[0..16]=attend, 나머지=masked │ self │
(첫 decode이므로 모든 17개 토큰이 window 내 → global과 동일)
```

#### ⑤ LLM Transformer 실행 (NPU)

```
llm.Run("decode",
        input:  {embeddings:       INT16 [1, 1, 1152],
                 pos_emb_cos:      INT16 [1, 1, 1, 256],
                 pos_emb_sin:      INT16 [1, 1, 1, 256],
                 pos_emb_local_cos: INT16 [1, 1, 1, 256],
                 pos_emb_local_sin: INT16 [1, 1, 1, 256],
                 mask_global:      INT16 [1, 1, 1, 1281],
                 mask_local:       INT16 [1, 1, 1, 1281],
                 kv_cache_k_0..25: INT16 [1, 1, 1280, 256] × 26,
                 kv_cache_v_0..25: INT16 [1, 1, 256, 1280] × 26},

        output: {kv_slice_k_0..25: INT16 [1, 1, 1, 256] × 26,
                 kv_slice_v_0..25: INT16 [1, 1, 256, 1] × 26,
                 logits:           INT16 [1, 1, 262144]})
```

**NPU Decode 내부** (26 layers, 단일 토큰):

```
Layer i (i = 0..25):
  x: [1, 1, 1152]  (단일 토큰)

  # Self-Attention
  Q = x × W_Q → [1, 4, 1, 256]     (4 query heads)
  K = x × W_K → [1, 1, 1, 256]     (1 KV head)
  V = x × W_V → [1, 1, 1, 256]

  Q = apply_rope(Q, pos_emb for position 17)
  K = apply_rope(K, pos_emb for position 17)

  K_full = kv_cache_k[i][:,:,:17,:] ++ K  → attend to positions 0~17
  V_full = kv_cache_v[i][:,:,:,:17] ++ V

  attn_weights = Q @ K_full.T / √256     → [1, 4, 1, 18]
  attn_weights = attn_weights × mask      → masked softmax
  attn_out = attn_weights @ V_full        → [1, 4, 1, 256]
  attn_out = reshape → [1, 1, 1024] → W_O → [1, 1, 1152]

  # FFN
  gate = x × W_gate → [1, 1, 6912]
  up   = x × W_up   → [1, 1, 6912]
  h = gelu(gate) ⊙ up → [1, 1, 6912]
  out = h × W_down   → [1, 1, 1152]

  output_kv_slice_k[i] = K  → [1, 1, 1, 256]
  output_kv_slice_v[i] = V  → [1, 1, 256, 1]

Final RMSNorm → Linear(head) → logits: [1, 1, 262144]
```

#### ⑥ Cache Update 실행 (CPU)

```
aux.Run("decode_cache_update",
        input:  {kv_cache_k_0..25:  [1, 1, 1280, 256] × 26,
                 kv_cache_v_0..25:  [1, 1, 256, 1280] × 26,
                 kv_slice_k_0..25:  [1, 1, 1, 256] × 26,
                 kv_slice_v_0..25:  [1, 1, 256, 1] × 26,
                 input_pos:         INT32 [1] = [17]},

        output: {kv_cache_k_0..25:  [1, 1, 1280, 256] × 26,    ← position 17 업데이트
                 kv_cache_v_0..25:  [1, 1, 256, 1280] × 26})
```

```
kv_cache_k_0 — After Decode Step 17:
  ┌───────────────────────────────────────────────────┐
  │ [0,0,0,:]:    K_layer0(token 0, <bos>)             │
  │ [0,0,1,:]:    K_layer0(token 1, <sot>)             │
  │ ...                                                 │
  │ [0,0,16,:]:   K_layer0(token 16, "model")          │
  │ [0,0,17,:]:   K_layer0(token 17, "\n")  ← ★ NEW    │
  │ [0,0,18,:]:   0 0 0 ... 0               ← 미사용   │
  │ ...                                                 │
  │ [0,0,1279,:]: 0 0 0 ... 0                          │
  └───────────────────────────────────────────────────┘
```

### 7.3 Greedy Sampling

```cpp
ApplyGreedySampling(decoded_logits)
  // logits: INT16 [1, 1, 262144]
  // → argmax over 262,144 values
  // → max_index = 651  (예시: "The" 토큰)
```

```
logits (INT16) [1, 1, 262144]:
┌──────┬──────┬──────┬──────┬─── ... ───┬──────┬─── ... ───┐
│ -120 │ -85  │ 102  │ -40  │   ...     │ 2847 │   ...     │
│ [0]  │ [1]  │ [2]  │ [3]  │           │ [651]│           │
└──────┴──────┴──────┴──────┴─── ... ───┴──────┴─── ... ───┘
                                          ↑ max → token 651 = "The"
```

### 7.4 후속 Decode Steps 반복

```
Decode Step 18 (2nd decode):
  pending_token = 651 ("The")
  ├── embedder: token 651 → INT16 [1,1,1152]
  ├── RoPE: position 18 → pos_emb cos/sin
  ├── Mask: step=18, attend to positions 0~18
  ├── LLM: decode with kv_cache[0..18 filled]
  │   → logits → argmax → 예: 19923 ("Bur")
  ├── Cache Update: position 18에 KV 삽입
  └── output: token 19923

Decode Step 19 (3rd decode):
  pending_token = 19923 ("Bur")
  ├── embedder: token 19923 → INT16 [1,1,1152]
  ├── RoPE: position 19
  ├── Mask: step=19, attend to positions 0~19
  ├── LLM: → logits → argmax → 예: 235297 ("j")
  ├── Cache Update: position 19
  └── output: token 235297

... (계속) ...

Decode Step N:
  ├── LLM: → logits → argmax → 107 (<end_of_turn>)
  └── stop_token 감지 → 생성 종료
```

### 7.5 KV Cache 진화 과정

```
Step  | KV Cache 유효 위치 | 새로 추가된 토큰
------+--------------------+------------------
 17   | 0..17              | "\n" (pending from prefill)
 18   | 0..18              | "The" (1st generated)
 19   | 0..19              | "Bur" (2nd generated)
 20   | 0..20              | "j"   (3rd generated)
 21   | 0..21              | " Khal" ...
 ...  | ...                | ...
 N    | 0..N               | <end_of_turn> → STOP
```

### 7.6 생성 예시 출력

```
The Burj Khalifa in Dubai, UAE, is the tallest building in the world,
standing at 828 meters (2,717 feet) tall.<end_of_turn>
```

(약 30~40 토큰 생성, 각 단계마다 위의 Decode 파이프라인 반복)

---

<a name="phase7"></a>
## 8. Phase 7: 응답 완성 및 벤치마크

### 8.1 Streaming Callback

매 Decode step마다 생성된 토큰은 `CreateMessageCallback()`을 통해 스트리밍 출력:

```cpp
// litert_lm_main.cc:72-90
// 각 decode step에서:
// 1. 토큰 ID → Detokenize → 텍스트
// 2. JsonMessage 생성: {"content": [{"text": "The"}]}
// 3. callback 호출 → std::cout << "The" << std::flush;
```

### 8.2 종료 조건

Decode 루프는 다음 조건에서 종료:
1. **Stop token 감지**: `token_id ∈ {1 (<eos>), 107 (<end_of_turn>)}`
2. **Max output tokens 도달**: `max_output_tokens` (기본 INT_MAX)
3. **KV cache 가득 참**: `current_step_ >= 1280`
4. **Timeout**: `engine->WaitUntilDone(10분)`

### 8.3 벤치마크 출력

```cpp
// litert_lm_main.cc:168-169
auto benchmark_info = conversation->GetBenchmarkInfo();
std::cout << std::endl << *benchmark_info << std::endl;
```

NPU executor의 LatencyStats 출력 예시:

```
====== PREFILL STATS ======
Total prefill latency [us]: 45000
(e2e) Prefill num tokens: 128
(e2e) Prefill tokens per second: 2844.4
(TransformerStackOnly) Prefill tokens per second: 3200.0
------ Prefill breakdown ------
Total prefill prepare input tensors latency [us]: 200 (0.44%)
Total prefill embedder inference latency [us]: 5000 (11.1%)
Total prefill rope inference latency [us]: 300 (0.67%)
Total prefill mask inference latency [us]: 800 (1.78%)
Total prefill LLM inference latency [us]: 35000 (77.8%)
Total prefill cache update inference latency [us]: 3700 (8.22%)

====== DECODE STATS ======
Total decode latency [us]: 600000
Decode num tokens: 35
Decode tokens per second: 58.3
(TransformerStackOnly) Decode tokens per second: 64.8
------ Decode breakdown ------
Total decode prepare input tensors latency [us]: 350 (0.06%)
Total decode embedder inference latency [us]: 17500 (2.92%)
Total decode rope inference latency [us]: 3500 (0.58%)
Total decode mask inference latency [us]: 7000 (1.17%)
Total decode LLM inference latency [us]: 540000 (90.0%)
Total decode cache update inference latency [us]: 28000 (4.67%)
Total decode sampling latency [us]: 3650 (0.61%)
```

### 8.4 Reset

Conversation/Session이 소멸될 때 executor의 `Reset()`이 호출:

```cpp
Reset()
  ├── PrintLatencyStats()    ← 벤치마크 출력
  ├── current_step_ = 0
  ├── processed_tokens_.RollBackToStep(0)
  ├── sampled_ids_.clear()
  ├── latency_stats_ = {}
  └── ClearKVCache()         ← 모든 KV cache를 0으로 초기화
```

---

<a name="summary"></a>
## 9. 전체 텐서 데이터 흐름 요약도

### 9.1 Prefill 데이터 흐름

```
User Prompt: "What is the tallest building in the world?"
                            │
                            ▼
                ┌────────────────────┐
                │    Tokenizer       │ (SentencePiece, CPU)
                │    vocab=262,144   │
                └────────┬───────────┘
                         │ token_ids: INT32 [1, 18]
                         │ [2, 106, 1645, 108, ...]
                         ▼
        ┌────────────────────────────────────────────────────┐
        │            PrefillInternal("prefill_128")           │
        │                                                     │
        │  ┌─────────────┐    ┌────────────┐                 │
        │  │token_ids     │    │input_pos    │                 │
        │  │INT32 [1,128] │    │INT32 [128]  │                 │
        │  │(17 tokens +  │    │[0,1,...,16,  │                 │
        │  │ 111 padding) │    │ 0,...,0]     │                 │
        │  └──────┬───────┘    └──────┬──────┘                 │
        │         │                   │                         │
        │         ▼                   ▼                         │
        │  ┌──────────────┐   ┌──────────────┐                │
        │  │  EMBEDDER    │   │    RoPE      │ (AUX CPU)      │
        │  │  (CPU)       │   │  prefill_    │                │
        │  │  INT4 weight │   │  rope_128    │                │
        │  │  [262K,1152] │   └──────┬───────┘                │
        │  └──────┬───────┘          │                         │
        │         │                  │ pos_emb_*:              │
        │         │ embeddings:      │ INT16 [1,128,1,256] ×4  │
        │         │ INT16            │                         │
        │         │ [1,128,1152]     │                         │
        │         │                  │  ┌──────────────┐       │
        │         │          step=0──┤  │    MASK      │       │
        │         │      token_ids──►│  │  prefill_    │       │
        │         │                  │  │  mask_128    │       │
        │         │                  │  └──────┬───────┘       │
        │         │                  │         │               │
        │         │                  │ mask_global/local:       │
        │         │                  │ INT16 [1,1,128,1408]     │
        │         ▼                  ▼                         │
        │  ┌──────────────────────────────────────────┐       │
        │  │         LLM TRANSFORMER (NPU/APU)         │       │
        │  │         "prefill_128" signature            │       │
        │  │         26 layers × 1 APU delegate         │       │
        │  │                                            │       │
        │  │  + kv_cache_k/v: INT16 [1,1,1280,256/1280]│       │
        │  │                  × 26 layers (초기 0)       │       │
        │  │                                            │       │
        │  │  → kv_slice_k: INT16 [1,1,128,256] × 26   │       │
        │  │  → kv_slice_v: INT16 [1,1,256,128] × 26   │       │
        │  └──────────────────────┬────────────────────┘       │
        │                         │                             │
        │                         ▼                             │
        │  ┌──────────────────────────────────────────┐        │
        │  │        CACHE UPDATE (AUX CPU)             │        │
        │  │  "prefill_cache_update_128"               │        │
        │  │                                            │        │
        │  │  kv_cache[pos 0..16] ← kv_slice[0..16]   │        │
        │  │  (positions 17~127 are padding, ignored)   │        │
        │  └──────────────────────────────────────────┘        │
        └────────────────────────────────────────────────────┘
```

### 9.2 Decode 데이터 흐름 (매 Step 반복)

```
        pending_token: INT32 = [651]  ("The")
        step: 18
                │
    ┌───────────┼───────────────────────────────────────┐
    │           ▼                                        │
    │  ┌──────────────┐  ┌──────────┐  ┌──────────┐    │
    │  │  EMBEDDER    │  │   RoPE   │  │   MASK   │    │
    │  │ decode_      │  │ decode_  │  │ decode_  │    │
    │  │ embedder     │  │ rope     │  │ mask     │    │
    │  │              │  │          │  │          │    │
    │  │ [651]→emb    │  │ [18]→    │  │ step=18  │    │
    │  │ INT16        │  │ cos/sin  │  │ → global │    │
    │  │ [1,1,1152]   │  │ INT16    │  │ + local  │    │
    │  │              │  │ [1,1,1,  │  │ INT16    │    │
    │  └──────┬───────┘  │ 256]×4   │  │ [1,1,1,  │    │
    │         │          └────┬─────┘  │ 1281]×2  │    │
    │         │               │        └────┬─────┘    │
    │         ▼               ▼             ▼          │
    │  ┌────────────────────────────────────────────┐  │
    │  │      LLM TRANSFORMER (NPU/APU)              │  │
    │  │      "decode" signature                     │  │
    │  │      26 layers × 1 APU delegate             │  │
    │  │                                              │  │
    │  │  + kv_cache: [1,1,1280,256] × 52            │  │
    │  │    (positions 0~17 filled from prefill+      │  │
    │  │     previous decodes)                        │  │
    │  │                                              │  │
    │  │  → kv_slice_k/v: [1,1,1,256] × 52          │  │
    │  │  → logits: INT16 [1, 1, 262144]             │  │
    │  └─────────────┬──────────────┬────────────────┘  │
    │                │              │                    │
    │                ▼              ▼                    │
    │  ┌──────────────────┐  ┌──────────────────────┐  │
    │  │  CACHE UPDATE    │  │  GREEDY SAMPLING     │  │
    │  │  decode_cache_   │  │                      │  │
    │  │  update          │  │  argmax(logits)      │  │
    │  │                  │  │  → token_id = 19923  │  │
    │  │  kv_cache[18]    │  │    ("Bur")           │  │
    │  │  ← kv_slice      │  │                      │  │
    │  └──────────────────┘  └──────────┬───────────┘  │
    │                                   │               │
    └───────────────────────────────────┼───────────────┘
                                        │
                                        ▼
                            next pending_token = 19923
                            output → callback → stdout
                            (repeat until stop token)
```

---

<a name="appendix"></a>
## 10. 부록: Gemma3-1B 모델 아키텍처 상수

### 10.1 모델 아키텍처 파라미터

| 파라미터 | 값 | 설명 |
|:---|---:|:---|
| `num_layers` | 26 | Transformer 레이어 수 |
| `hidden_dim` / `embedding_dim` | 1,152 | 은닉 차원 |
| `num_attention_heads` | 4 | Query head 수 |
| `num_kv_heads` | 1 | KV head 수 (GQA) |
| `head_dim` | 256 | 각 head의 차원 (= hidden_dim / num_heads ≈) |
| `intermediate_size` | 6,912 | FFN 확장 차원 |
| `vocab_size` | 262,144 | 어휘 크기 (SentencePiece) |
| `sliding_window_size` | 512 | Local attention window 크기 |
| `max_position_embeddings` | 32,768 | 전체 최대 위치 (RoPE base) |

### 10.2 NPU 모델 특화 파라미터

| 파라미터 | 값 | 설명 |
|:---|---:|:---|
| `max_kv_len` (ekv) | 1,280 | KV cache 최대 시퀀스 길이 |
| `prefill_chunk_size` | 128 | 한 번에 prefill 가능한 토큰 수 |
| `weight_quantization` | INT4 | 모델 가중치 양자화 |
| `activation_type` | INT16 | 활성화(중간값) 데이터 타입 |
| `logits_type` | INT16 | 출력 logits 데이터 타입 |
| `embedding_weight` | INT4 [262144, 1152] | 임베딩 테이블 크기 |
| `kv_cache_k shape` | [1, 1, 1280, 256] | K cache per layer |
| `kv_cache_v shape` | [1, 1, 256, 1280] | V cache per layer (transposed) |

### 10.3 텐서 형상 총정리

#### Prefill 시그니처 (`prefill_128`)

| 텐서 | 방향 | dtype | shape | 개수 |
|:---|:---|:---|:---|---:|
| `embeddings` | input | INT16 | [1, 128, 1152] | 1 |
| `pos_emb_cos` | input | INT16 | [1, 128, 1, 256] | 1 |
| `pos_emb_sin` | input | INT16 | [1, 128, 1, 256] | 1 |
| `pos_emb_local_cos` | input | INT16 | [1, 128, 1, 256] | 1 |
| `pos_emb_local_sin` | input | INT16 | [1, 128, 1, 256] | 1 |
| `mask_global` | input | INT16 | [1, 1, 128, 1408] | 1 |
| `mask_local` | input | INT16 | [1, 1, 128, 1408] | 1 |
| `kv_cache_k_N` | input | INT16 | [1, 1, 1280, 256] | 26 |
| `kv_cache_v_N` | input | INT16 | [1, 1, 256, 1280] | 26 |
| `kv_slice_k_N` | output | INT16 | [1, 1, 128, 256] | 26 |
| `kv_slice_v_N` | output | INT16 | [1, 1, 256, 128] | 26 |
| **합계** | | | | **입력 59, 출력 52** |

#### Decode 시그니처 (`decode`)

| 텐서 | 방향 | dtype | shape | 개수 |
|:---|:---|:---|:---|---:|
| `embeddings` | input | INT16 | [1, 1, 1152] | 1 |
| `pos_emb_cos` | input | INT16 | [1, 1, 1, 256] | 1 |
| `pos_emb_sin` | input | INT16 | [1, 1, 1, 256] | 1 |
| `pos_emb_local_cos` | input | INT16 | [1, 1, 1, 256] | 1 |
| `pos_emb_local_sin` | input | INT16 | [1, 1, 1, 256] | 1 |
| `mask_global` | input | INT16 | [1, 1, 1, 1281] | 1 |
| `mask_local` | input | INT16 | [1, 1, 1, 1281] | 1 |
| `kv_cache_k_N` | input | INT16 | [1, 1, 1280, 256] | 26 |
| `kv_cache_v_N` | input | INT16 | [1, 1, 256, 1280] | 26 |
| `kv_slice_k_N` | output | INT16 | [1, 1, 1, 256] | 26 |
| `kv_slice_v_N` | output | INT16 | [1, 1, 256, 1] | 26 |
| `logits` | output | INT16 | [1, 1, 262144] | 1 |
| **합계** | | | | **입력 59, 출력 53** |

### 10.4 메모리 사용량 추정 (KV Cache)

```
KV Cache per layer:
  K: 1 × 1 × 1280 × 256 × 2 bytes (INT16) = 655,360 bytes = 640 KB
  V: 1 × 1 × 256 × 1280 × 2 bytes (INT16) = 655,360 bytes = 640 KB
  Per layer total: 1.25 MB

Total KV cache (26 layers):
  26 × 1.25 MB = 32.5 MB
```

### 10.5 Sub-model 구성 및 실행 하드웨어

| Sub-model | 파일 크기 | 실행 HW | 역할 |
|:---|---:|:---|:---|
| `TF_LITE_EMBEDDER` | 150 MB | CPU | Token ID → INT16 embedding lookup |
| `TF_LITE_AUX` | 116 KB | CPU | RoPE, Attention Mask, KV Cache Update |
| `TF_LITE_PREFILL_DECODE` | 830 MB | NPU (APU) | 26-layer Transformer (1 delegate op) |

### 10.6 파일 이름 해석

```
Gemma3-1B-IT_q4_ekv1280_mt6989.litertlm
│       │  │  │  │       │      └── LiteRT-LM 컨테이너 포맷
│       │  │  │  │       └── MediaTek Dimensity 9300 타겟
│       │  │  │  └── KV cache 최대 길이 1280
│       │  │  └── External KV cache
│       │  └── INT4 weight quantization
│       └── Instruction-Tuned (chat model)
└── Gemma 3, 1 Billion parameters
```
