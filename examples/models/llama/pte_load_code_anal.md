# ExecuTorch .PTE 파일 로딩/파싱 및 Runner 적용 상세 분석서

> **가정 (예시 값)**
> - 모델: LLaMA 3 8B Instruct (XNNPACK 백엔드)
> - 파일: `llama3-8b.pte` (약 4.5GB)
> - 토크나이저: `tokenizer.model` (TikToken, LLaMA 3)
> - vocab_size: 128000, hidden_size: 4096, num_layers: 32
> - max_seq_len: 2048, use_kv_cache: true
> - 프롬프트: `"The answer to the ultimate question is"`
> - 토큰: `[128000, 791, 4320, 311, 279, 17139, 3488, 374]` (BOS 포함, 8개)

---

## 목차

1. [.PTE 파일 바이너리 구조](#1-pte-파일-바이너리-구조)
2. [FlatBuffer 스키마 (program.fbs)](#2-flatbuffer-스키마-programfbs)
3. [Module 생성 및 DataLoader](#3-module-생성-및-dataloader)
4. [Program::load() — .PTE 파싱](#4-programload--pte-파싱)
5. [메타데이터 메서드 실행](#5-메타데이터-메서드-실행)
6. [Module::load_method("forward") — 메서드 로드](#6-moduleload_methodforward--메서드-로드)
7. [Method::init() — 값 파싱, 텐서 할당, Delegate 초기화](#7-methodinit--값-파싱-텐서-할당-delegate-초기화)
8. [Module::execute() — 추론 실행](#8-moduleexecute--추론-실행)
9. [Method::execute() — 명령어 실행 루프](#9-methodexecute--명령어-실행-루프)
10. [Runner에서 .PTE 필드가 적용되는 지점 총정리](#10-runner에서-pte-필드가-적용되는-지점-총정리)
11. [TikToken 토크나이저 상세 분석](#11-tiktoken-토크나이저-상세-분석)

---

## 1. .PTE 파일 바이너리 구조

### 1.1 파일 레이아웃

```
┌──────────────────────────────────────────────────────────┐
│ Extended Header (64 bytes)                               │  ← offset 0
│   magic: "eh00"                                          │
│   program_size: uint64  (FlatBuffer 영역 크기)           │
│   segment_base_offset: uint64 (세그먼트 시작 위치)       │
│   segment_data_size: uint64 (세그먼트 총 크기)           │
├──────────────────────────────────────────────────────────┤
│ FlatBuffer Program Data                                  │  ← offset 64
│   ┌─ file_identifier: "ET12" (매직 넘버)                │
│   ├─ root_type: Program                                  │
│   │   ├── version: 0                                     │
│   │   ├── execution_plan: [...]                          │  ← 메서드별 실행 계획
│   │   ├── constant_buffer: [] (비어있음, deprecated)     │
│   │   ├── backend_delegate_data: [...]                   │  ← XNNPACK 바이너리 등
│   │   ├── segments: [DataSegment, ...]                   │  ← 세그먼트 오프셋/크기
│   │   ├── constant_segment: SubsegmentOffsets            │  ← 상수 텐서 위치
│   │   └── named_data: [...]                              │  ← 명명된 외부 데이터
│   └──────────────────────────────────────────────────────┘
├──────────────────────────────────────────────────────────┤
│ Segment 0: 상수 데이터 (가중치)                          │  ← segment_base_offset
│   ┌─ 텐서 0 데이터 (16바이트 정렬)                      │
│   ├─ 텐서 1 데이터                                      │
│   ├─ ... (수천 개의 가중치 텐서)                         │
│   └─ 텐서 N 데이터                                      │
├──────────────────────────────────────────────────────────┤
│ Segment 1: XNNPACK Delegate 데이터 (선택적)              │
│   └─ XNNPACK이 컴파일한 연산 그래프 바이너리             │
├──────────────────────────────────────────────────────────┤
│ (추가 세그먼트...)                                       │
└──────────────────────────────────────────────────────────┘
```

### 1.2 크기 예시 (LLaMA 3 8B, INT4 양자화)

```
Extended Header:         64 bytes
FlatBuffer Program:      ~50 MB (실행 계획, 메타데이터, 구조 정보)
Segment 0 (가중치):      ~4.3 GB (INT4 양자화 가중치)
Segment 1 (XNNPACK):     ~10 MB (delegate 바이너리)
────────────────────────
총 파일 크기:            ~4.4 GB
```

### 1.3 매직 넘버 검증

```
파일의 FlatBuffer 영역에서 offset +4~+7 위치에 "ET12" 존재:
  bytes[4] = 'E' (0x45)
  bytes[5] = 'T' (0x54)
  bytes[6] = '1' (0x31)
  bytes[7] = '2' (0x32)

이 매직이 맞지 않으면 Program::load()가 실패
```

---

## 2. FlatBuffer 스키마 (program.fbs)

**파일**: `schema/program.fbs`

### 2.1 핵심 테이블 구조

```
Program (루트)
├── version: uint
├── execution_plan: [ExecutionPlan]      ← ★ 각 메서드의 실행 계획
│   ├── [0] "forward"                    ← LLM 추론 메서드
│   ├── [1] "get_max_seq_len"            ← 메타데이터 메서드
│   ├── [2] "get_max_context_len"
│   ├── [3] "use_kv_cache"
│   ├── [4] "enable_dynamic_shape"
│   ├── [5] "get_eos_ids"
│   └── ... (기타 메타데이터)
├── constant_buffer: [Buffer]            ← (deprecated, 보통 비어있음)
├── backend_delegate_data: [BackendDelegateInlineData]
│   └── [0] XNNPACK delegate 바이너리
├── segments: [DataSegment]              ← 세그먼트 위치 목록
│   ├── [0] {offset: 0, size: 4500000000}  ← 상수(가중치) 세그먼트
│   └── [1] {offset: 4500000000, size: 10000000}  ← delegate 세그먼트
├── constant_segment: SubsegmentOffsets  ← 각 상수 텐서의 세그먼트 내 오프셋
│   ├── segment_index: 0
│   └── offsets: [0, 0, 16384, 32768, ...]  ← offsets[0]은 예약됨
└── named_data: [NamedData]              ← 이름으로 참조할 수 있는 데이터
```

### 2.2 ExecutionPlan 상세

```
ExecutionPlan ("forward")
├── name: "forward"
├── container_meta_type: ContainerMetadata
│   ├── encoded_inp_str: "..." (PyTree 입력 구조)
│   └── encoded_out_str: "..." (PyTree 출력 구조)
├── values: [EValue × ~5000개]           ← ★ 이 메서드가 사용하는 모든 값
│   ├── [0] Tensor {sizes:[1,8], dtype:Long}     ← input token_ids
│   ├── [1] Tensor {sizes:[1], dtype:Long}        ← input cache_position
│   ├── [2] Tensor {sizes:[1,8,128000], dtype:Float} ← output logits
│   ├── [3] Tensor {weight, const, dtype:Int4}    ← layer0.attention.wq
│   ├── [4] Tensor {weight, const, dtype:Int4}    ← layer0.attention.wk
│   ├── ... (수천 개의 텐서: 가중치, 활성화, KV 캐시)
│   ├── [4999] Int {value: 32}                     ← num_heads 상수
│   └── [5000] Bool {value: true}                  ← 기타 플래그
├── inputs: [0, 1]                       ← values[0], values[1]이 입력
├── outputs: [2]                         ← values[2]가 출력
├── chains: [Chain]
│   └── [0] Chain
│       └── instructions: [Instruction × ~2개]
│           ├── [0] DelegateCall {delegate_index:0, args:[0,1,2,...,5000]}
│           │       ← XNNPACK가 대부분의 연산을 처리
│           └── [1] (후처리 커널, 있다면)
├── operators: [Operator]                ← 커널 연산자 목록
│   └── (XNNPACK으로 전부 위임되면 비어있을 수 있음)
├── delegates: [BackendDelegate]
│   └── [0] BackendDelegate
│       ├── id: "XnnpackBackend"         ← ★ XNNPACK 백엔드 식별자
│       ├── processed: {location:SEGMENT, index:1}
│       │   ← Segment 1에 저장된 XNNPACK 컴파일 바이너리
│       └── compile_specs: [CompileSpec]
│           └── (컴파일 옵션)
└── non_const_buffer_sizes: [int64]      ← 메모리 풀 크기
    ├── [0] 0                            ← 상수용 (런타임에서 무시)
    ├── [1] 52428800                     ← 활성화 메모리 (~50MB)
    └── [2] 0                            ← 공유 버퍼 (없으면 0)
```

### 2.3 Tensor의 4가지 분류

스키마에서 `data_buffer_idx`와 `allocation_info`의 조합으로 텐서 종류를 결정합니다:

| data_buffer_idx | allocation_info | 분류 | 설명 | 예시 |
|----------------|-----------------|------|------|------|
| > 0 | null | **상수 (Constant)** | 가중치 등, 변경 불가 | `wq`, `wk`, `wv`, `wo` |
| 0 | non-null | **변경 가능 (Mutable)** | 런타임 메모리 할당 | KV 캐시, 활성화 텐서 |
| 0 | null | **입력/플레이스홀더** | 실행 시 외부에서 제공 | `token_ids`, `cache_position` |
| > 0 | non-null | **변경 가능 + 초기값** | 메모리 할당 + 초기 데이터 | 온디바이스 학습 가중치 |

```
예시: forward 메서드의 values 배열

values[0]: Input token_ids
  scalar_type: Long (int64)
  sizes: [1, 8]                   ← 동적 shape (prefill 시 변경)
  data_buffer_idx: 0              ← 데이터 없음
  allocation_info: null           ← 메모리 미할당
  shape_dynamism: DYNAMIC_BOUND   ← 동적이나 상한 있음
  → 분류: 입력/플레이스홀더 — 실행 시 set_input()으로 제공

values[1]: Cache position
  scalar_type: Long
  sizes: [1]
  data_buffer_idx: 0
  allocation_info: null
  → 분류: 입력/플레이스홀더

values[2]: Output logits
  scalar_type: Float
  sizes: [1, 8, 128000]
  data_buffer_idx: 0
  allocation_info: {memory_id:1, offset:0}
  → 분류: Mutable — HierarchicalAllocator에서 메모리 할당

values[3]: layer0.attention.wq (가중치)
  scalar_type: Int4 (양자화)
  sizes: [4096, 4096]
  data_buffer_idx: 17             ← constant_segment.offsets[17]에서 위치 참조
  allocation_info: null
  → 분류: 상수 — 세그먼트 데이터를 직접 포인팅 (복사 없음)

values[100]: layer0.kv_cache
  scalar_type: Float
  sizes: [1, 32, 2048, 128]      ← [batch, heads, max_seq, head_dim]
  data_buffer_idx: 0
  allocation_info: {memory_id:1, offset:524288}
  → 분류: Mutable — 활성화 메모리에서 할당
```

---

## 3. Module 생성 및 DataLoader

**파일**: `extension/module/module.cpp`

### 3.1 Module 생성

```cpp
// create_text_llm_runner()에서 호출
auto module = std::make_unique<Module>(
    "llama3-8b.pte",                       // file_path
    Module::LoadMode::MmapUseMlockIgnoreErrors,  // load_mode
    nullptr                                 // event_tracer
);
```

### 3.2 DataLoader 생성 (Module::load_internal)

```
Module::load_internal()
│
├── [LoadMode에 따른 DataLoader 생성]
│   MmapUseMlockIgnoreErrors:
│     MmapDataLoader::from("llama3-8b.pte", UseMlockIgnoreErrors)
│     → 파일을 mmap()으로 메모리에 매핑
│     → mlock()으로 페이지 고정 시도 (실패해도 계속 진행)
│
│   실제 동작:
│     fd = open("llama3-8b.pte", O_RDONLY)
│     file_size = 4,500,000,064 bytes
│     addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
│     mlock(addr, file_size)  // 실패해도 OK
│     → DataLoader가 이 매핑된 주소를 통해 데이터 제공
│
├── [외부 데이터 파일 로드 (있으면)]
│   data_files_가 비어있지 않으면:
│     각 .ptd 파일에 대해 FlatTensorDataMap 생성
│     → MergedDataMap으로 통합
│
└── [Program 로드]
    Program::load(data_loader_.get(), Minimal)
    → 아래 섹션 4에서 상세 설명
```

### 3.3 LoadMode별 동작 차이

| LoadMode | 동작 | 메모리 사용 | 지연 |
|----------|------|------------|------|
| `File` | 전체 파일을 RAM에 읽기 | 높음 (파일 크기만큼) | 높음 (전체 읽기) |
| `Mmap` | mmap만 (mlock 안 함) | 낮음 (접근한 페이지만) | 낮음 (lazy) |
| `MmapUseMlock` | mmap + mlock (실패 시 에러) | 높음 (전체 고정) | 중간 |
| `MmapUseMlockIgnoreErrors` | mmap + mlock (실패 무시) | 가변적 | 낮음 ✅ |

**LLM Runner의 기본값은 `MmapUseMlockIgnoreErrors`**:
- mmap으로 4.5GB 파일을 가상 메모리에 매핑 (물리 메모리 즉시 사용 안 함)
- 페이지 접근 시 on-demand로 디스크에서 로드
- mlock 성공하면 페이지 스왑 방지 (추론 지연 감소)
- mlock 실패해도 정상 동작 (메모리 부족 환경에서 graceful)

---

## 4. Program::load() — .PTE 파싱

**파일**: `runtime/executor/program.cpp`

```
Program::load(data_loader, Minimal)
│
├── [단계 1] Extended Header 읽기
│   data_loader->load(offset=0, size=64) → 64바이트 버퍼
│   ExtendedHeader 파싱:
│     magic = "eh00" ✓
│     program_size = 52,428,800 (50MB)      ← FlatBuffer 영역 크기
│     segment_base_offset = 52,428,864      ← 세그먼트 시작 (Header+FB 크기)
│     segment_data_size = 4,447,571,200     ← 세그먼트 총 크기 (~4.1GB)
│
├── [단계 2] FlatBuffer Program 데이터 로드
│   data_loader->load(offset=64, size=52,428,800)
│   → 50MB의 FlatBuffer 데이터를 FreeableBuffer로 로드
│   (mmap 모드에서는 실제로 메모리 복사 없이 포인터만 얻음)
│
├── [단계 3] 매직 넘버 검증
│   flatbuffers::Verify(program_data, "ET12")
│   → FlatBuffer의 file_identifier가 "ET12"인지 확인
│   → 실패하면 Error::InvalidProgram 반환
│
├── [단계 4] Minimal 검증
│   Verification::Minimal 모드:
│     → 헤더 매직만 확인 (빠름)
│   Verification::InternalConsistency 모드 (선택적):
│     → FlatBuffer 전체 구조 검증
│     → 텐서 numel 오버플로 체크
│     → 리스트 원소 타입 검증
│
├── [단계 5] FlatBuffer 디시리얼라이즈
│   internal_program_ = GetProgram(program_data)
│   → FlatBuffer의 루트 테이블 포인터 획득
│   → ★ 이 시점에서 .pte 내의 모든 구조에 접근 가능
│   → 실제로는 제로카피 — 원본 바이트 배열을 구조체처럼 읽음
│
├── [단계 6] Constant Segment 로드
│   if (internal_program_->constant_segment()->offsets()->size() > 1):
│     segment_index = constant_segment.segment_index  (예: 0)
│     segment = segments[0]  → {offset: 0, size: 4,300,000,000}
│     actual_offset = segment_base_offset + segment.offset
│                   = 52,428,864 + 0 = 52,428,864
│     data_loader->load(offset=52,428,864, size=4,300,000,000)
│     → constant_segment_data_ 에 저장
│     (mmap이면 가상 메모리 매핑만, 실제 디스크 I/O는 접근 시)
│
└── [단계 7] NamedDataMap 생성 (있으면)
    internal_program_->named_data()가 비어있지 않으면:
      PteDataMap 생성 → 키-세그먼트 인덱스 매핑

반환: Program 객체
  ├── program_data_: FlatBuffer 바이트 (또는 mmap 포인터)
  ├── loader_: DataLoader* (세그먼트 추가 로드용)
  ├── internal_program_: FlatBuffer Program* (디시리얼라이즈된 루트)
  ├── segment_base_offset_: 52,428,864
  └── constant_segment_data_: 가중치 세그먼트 (mmap 포인터)
```

---

## 5. 메타데이터 메서드 실행

**파일**: `extension/llm/runner/llm_runner_helper.cpp`

`get_llm_metadata()`에서 모델의 메타데이터 메서드를 실행합니다. 이 메서드들은 .pte 파일 안에 별도의 ExecutionPlan으로 직렬화되어 있습니다.

```
.pte 파일의 execution_plan 목록:
  [0] "forward"              ← 메인 추론 메서드
  [1] "get_max_seq_len"      ← 메타데이터: 최대 시퀀스 길이
  [2] "get_max_context_len"  ← 메타데이터: 최대 컨텍스트 길이
  [3] "use_kv_cache"         ← 메타데이터: KV 캐시 사용 여부
  [4] "enable_dynamic_shape" ← 메타데이터: 동적 shape 사용 여부
  [5] "use_sdpa_with_kv_cache" ← 메타데이터: SDPA+KV캐시
  [6] "get_eos_ids"          ← 메타데이터: EOS 토큰 ID 목록
```

### 메타데이터 메서드의 ExecutionPlan 예시

```
ExecutionPlan ("get_max_seq_len")
├── name: "get_max_seq_len"
├── values: [EValue × 1]
│   └── [0] Int {value: 2048}       ← 상수값이 직접 직렬화됨
├── inputs: []                       ← 입력 없음
├── outputs: [0]                     ← values[0]이 출력
├── chains: [Chain]
│   └── instructions: []             ← 명령어 없음! (상수 반환만)
├── operators: []
├── delegates: []
└── non_const_buffer_sizes: [0]
```

### 실행 흐름

```
module->get("get_max_seq_len")
│
├── module->execute("get_max_seq_len", {})  ← 입력 없이 실행
│   ├── load_method("get_max_seq_len")
│   │   → ExecutionPlan 파싱, values[0] = Int(2048) 설정
│   │   → 명령어가 없으므로 초기값이 곧 출력
│   ├── method->execute()
│   │   → chains[0].instructions가 비어있음 → 아무것도 안 함
│   └── method->get_outputs()
│       → values[outputs[0]] = values[0] = Int(2048)
│
└── 반환: EValue(Int, 2048) → .toScalar().to<int64_t>() → 2048

이런 식으로 각 메타데이터를 추출:
  get_max_seq_len      → 2048
  get_max_context_len  → 2048
  use_kv_cache         → 1 (true)
  enable_dynamic_shape → 1 (true)
  use_sdpa_with_kv_cache → 0 (false)

get_eos_ids는 약간 다름:
  module->execute("get_eos_ids") → [EValue(Int, 128001), EValue(Int, 128009)]
  → eos_ids = {128001, 128009}  (end_of_text, eot_id)
```

---

## 6. Module::load_method("forward") — 메서드 로드

**파일**: `extension/module/module.cpp:353`

```
Module::load_method("forward")
│
├── [1] Program 로드 확인 (이미 로드됨)
│
├── [2] 메모리 계획 크기 조회
│   MethodMeta meta = program_->method_meta("forward")
│   meta.num_memory_planned_buffers() → 3
│   meta.memory_planned_buffer_size(0) → 0         (상수, 무시)
│   meta.memory_planned_buffer_size(1) → 52428800  (활성화, ~50MB)
│   meta.memory_planned_buffer_size(2) → 0         (공유, 사용 안 함)
│
│   MethodMeta는 ExecutionPlan의 non_const_buffer_sizes를 읽음:
│     non_const_buffer_sizes: [0, 52428800, 0]
│     → mem_id=1에 50MB 필요 (KV 캐시, 활성화 텐서 등)
│
├── [3] PlannedMemory 생성
│   buffer_sizes = [0, 52428800, 0]
│   planned_buffers[0] = vector<uint8_t>(0)        → 0바이트
│   planned_buffers[1] = vector<uint8_t>(52428800)  → 50MB 할당 ★
│   planned_buffers[2] = vector<uint8_t>(0)        → 0바이트
│
│   planned_spans[0] = Span(buffers[0].data(), 0)
│   planned_spans[1] = Span(buffers[1].data(), 52428800)
│   planned_spans[2] = Span(buffers[2].data(), 0)
│
│   HierarchicalAllocator(planned_spans)
│   → 3개 메모리 풀을 관리하는 계층적 할당자
│   → mem_id=1에서 오프셋 기반으로 텐서 메모리 할당
│
├── [4] MemoryManager 생성
│   MemoryManager(
│     memory_allocator = MallocMemoryAllocator,   ← 일반 메모리
│     planned_memory = HierarchicalAllocator,      ← 계획된 메모리
│     temp_allocator = MallocMemoryAllocator        ← 임시 메모리
│   )
│
├── [5] Program::load_method() 호출
│   program_->load_method(
│     "forward",
│     memory_manager,
│     event_tracer,
│     merged_data_map,
│     backend_options
│   )
│   → Method 객체 반환 (다음 섹션에서 상세 설명)
│
└── [6] 캐싱
    methods_["forward"] = MethodHolder {
      planned_memory,
      memory_manager,
      method
    }
    → 이후 execute() 호출 시 재사용
```

---

## 7. Method::init() — 값 파싱, 텐서 할당, Delegate 초기화

**파일**: `runtime/executor/method.cpp`

### 7.1 값 파싱 (parse_values)

```
Method::init()
├── parse_values()
│   ExecutionPlan.values 배열 (예: 5000개 EValue)를 순회하며 파싱
│
│   for (i = 0; i < 5000; i++):
│     flatbuffer_value = execution_plan->values[i]
│
│     switch (flatbuffer_value.type):
│
│       case Tensor:
│         parseTensor(flatbuffer_value.tensor, i)
│         → 아래 7.2에서 상세 설명
│
│       case Int:
│         values_[i] = EValue(flatbuffer_value.int_val)
│         예: values_[4999] = EValue(Int, 32)  ← num_heads
│
│       case Bool:
│         values_[i] = EValue(flatbuffer_value.bool_val)
│
│       case Double:
│         values_[i] = EValue(flatbuffer_value.double_val)
│
│       case IntList:
│         items = flatbuffer_value.items  → [1, 8, 128000]
│         values_[i] = EValue(IntList, items)
│
│       case TensorList:
│         item_indices = flatbuffer_value.items  → [100, 101, 102, ...]
│         → values_[100], values_[101], ... 참조하는 텐서 리스트 생성
```

### 7.2 텐서 파싱 (parseTensor) — 4가지 경우

```
parseTensor(fb_tensor, value_index):

[경우 1] 상수 텐서 (가중치)
  조건: data_buffer_idx > 0, allocation_info == null
  예: layer0.attention.wq (data_buffer_idx=17)

  data_ptr = program->get_constant_buffer_data(17, nbytes)
  내부:
    offset = constant_segment.offsets[17]  → 예: 1,048,576
    data_ptr = constant_segment_data_.data() + offset
    → mmap된 파일의 가중치 데이터를 직접 포인팅 (제로카피!)

  Tensor 생성:
    sizes = [4096, 4096]
    scalar_type = Int4 (UInt4)
    dim_order = [0, 1]
    data = data_ptr  ← 파일에서 직접 매핑된 포인터
    → 메모리 복사 없음! 디스크의 가중치를 직접 사용


[경우 2] Mutable 텐서 (활성화, KV 캐시)
  조건: data_buffer_idx == 0, allocation_info != null
  예: KV cache (allocation_info.memory_id=1, offset=524288)

  data_ptr = planned_memory->get(memory_id=1, offset=524288, nbytes)
  → HierarchicalAllocator가 mem_id=1의 50MB 풀에서
    오프셋 524288부터 필요한 크기만큼 반환
  → 이 메모리는 Module::load_method()에서 미리 할당됨

  Tensor 생성:
    sizes = [1, 32, 2048, 128]   ← [batch, heads, max_seq, head_dim]
    scalar_type = Float
    data = planned_memory_base + 524288


[경우 3] 입력/플레이스홀더 텐서
  조건: data_buffer_idx == 0, allocation_info == null
  예: token_ids (inputs[0])

  data_ptr = nullptr  ← 데이터 없음!
  → 실행 시 set_input()으로 외부에서 제공해야 함

  Tensor 생성:
    sizes = [1, 8]          ← 상한 shape (DYNAMIC_BOUND)
    scalar_type = Long
    data = nullptr


[경우 4] 초기값이 있는 Mutable 텐서
  조건: data_buffer_idx > 0, allocation_info != null
  예: 온디바이스 학습용 가중치

  (a) planned_memory에서 메모리 할당
  (b) mutable_data_segments에서 초기 데이터 로드
  (c) 할당된 메모리에 초기 데이터 복사
```

### 7.3 Delegate 초기화 (XNNPACK)

```
[Delegate 초기화]
for (d = 0; d < execution_plan.delegates.size(); d++):
  fb_delegate = execution_plan.delegates[d]

  [1] 백엔드 식별
  backend_id = fb_delegate.id  → "XnnpackBackend"
  backend = get_backend_class(backend_id)
  → 컴파일 시 등록된 XNNPACK 백엔드 구현체 반환

  [2] Delegate 데이터 로드
  processed_ref = fb_delegate.processed
  if processed_ref.location == SEGMENT:
    segment = segments[processed_ref.index]  → segments[1]
    offset = segment_base_offset + segment.offset
    data = data_loader->load(offset, segment.size)
    → XNNPACK이 컴파일한 바이너리 (~10MB) 로드
  elif processed_ref.location == INLINE:
    data = backend_delegate_data[processed_ref.index].data
    → FlatBuffer 내 인라인 데이터 사용

  [3] 백엔드 초기화
  BackendInitContext ctx(
    memory_allocator,
    event_tracer,
    "forward",
    named_data_map
  )
  delegate_handle = backend->init(ctx, processed_data, compile_specs)

  XNNPACK init 내부:
    → 바이너리에서 연산 그래프 역직렬화
    → XNNPACK 런타임 생성
    → 가중치 텐서 포인터 바인딩 (values_의 상수 텐서들)
    → 실행 계획 최적화 (연산 융합, 메모리 최적화)
    → 스레드풀 설정
    → delegate_handle 반환

  delegates_[d] = {backend, delegate_handle, processed_data}
```

### 7.4 명령어 해석 (Instruction Resolution)

```
[명령어 해석]
for each chain in execution_plan.chains:
  for each instruction in chain.instructions:

    switch (instruction.type):

      case DelegateCall:
        delegate_index = 0  ← XNNPACK
        args = [0, 1, 2, ..., 5000]  ← values[] 인덱스들
        → 실행 시 delegates_[0].execute(args) 호출

      case KernelCall:
        op_index = instruction.op_index
        operator = execution_plan.operators[op_index]
        → operator.name = "aten::add.out"
        → operator.overload = "out"
        → OperatorRegistry에서 함수 포인터 찾기
        op_fn = registry.lookup("aten::add.out")
        args = instruction.args  ← values[] 인덱스들

      case MoveCall:
        move_from, move_to 인덱스 저장

      case FreeCall:
        value_index 저장 (메모리 재사용용)
```

---

## 8. Module::execute() — 추론 실행

**파일**: `extension/module/module.cpp:423`

```
Module::execute("forward", inputs)
│
│ inputs = [
│   EValue(Tensor, tokens [1,8] Long),     ← [128000, 791, 4320, ...]
│   EValue(Tensor, cache_pos [1] Long)      ← [0]
│ ]
│
├── [1] 메서드 로드 확인 (이미 로드됨)
│   methods_["forward"].method 사용
│
├── [2] 입력 설정
│   for (index = 0; index < 2; index++):
│     method->set_input(inputs[index], index)
│
│   set_input(tokens_tensor, 0):
│     input_index = execution_plan.inputs[0]  → 0
│     values_[0] = tokens_tensor
│     → values_[0]의 Tensor에 데이터 포인터 설정
│     → 외부 토큰 데이터([128000, 791, ...])를 values_[0]에 연결
│     input_set_[0] = true
│
│   set_input(cache_pos_tensor, 1):
│     input_index = execution_plan.inputs[1]  → 1
│     values_[1] = cache_pos_tensor
│     input_set_[1] = true
│
├── [3] 실행
│   method->execute()
│   → 아래 섹션 9에서 상세 설명
│
├── [4] 출력 수집
│   outputs_size = method->outputs_size()  → 1
│   outputs = vector<EValue>(1)
│   method->get_outputs(outputs.data(), 1)
│     output_index = execution_plan.outputs[0]  → 2
│     outputs[0] = values_[2]  ← logits 텐서
│     → Tensor(shape=[1,8,128000], dtype=Float)
│
└── 반환: [EValue(Tensor, logits)]
```

---

## 9. Method::execute() — 명령어 실행 루프

```
Method::execute()
│
├── [검증]
│   is_initialized? → true ✓
│   all inputs set? → input_set_[0]=true, input_set_[1]=true ✓
│
├── [체인 실행]
│   for each chain (보통 1개):
│     for each instruction (보통 1~2개):
│
│       ─── Instruction 0: DelegateCall ───
│       │
│       │ delegate_index = 0 → XNNPACK
│       │ args = [values_[0], values_[1], values_[2], ..., values_[5000]]
│       │   → values_[0]: 입력 tokens [1,8] = [128000, 791, ...]
│       │   → values_[1]: cache_pos [1] = [0]
│       │   → values_[3~4998]: 가중치들 (mmap 포인터)
│       │   → values_[2]: 출력 logits 버퍼 (50MB 풀에서 할당됨)
│       │   → values_[100~]: KV 캐시 (50MB 풀에서 할당됨)
│       │
│       │ BackendExecutionContext ctx(event_tracer)
│       │ delegates_[0].execute(ctx, delegate_handle, args)
│       │
│       │ XNNPACK execute 내부:
│       │   → args에서 입력 텐서 추출
│       │   → 가중치 포인터 확인 (mmap에서 직접 읽기)
│       │   → XNNPACK 런타임 실행:
│       │     ├── Token Embedding
│       │     ├── 32 × Transformer Layer:
│       │     │   ├── RMSNorm
│       │     │   ├── Self-Attention (Q,K,V projection + SDPA + Output)
│       │     │   │   └── KV 캐시 읽기/쓰기 (values_[100~]에 직접 접근)
│       │     │   ├── RMSNorm
│       │     │   └── FFN (gate + up + SwiGLU + down)
│       │     ├── Final RMSNorm
│       │     └── Output Projection → logits
│       │   → 결과를 values_[2] (logits)에 기록
│       │
│       │ → values_[2]에 logits [1,8,128000] 기록 완료
│       │
│       ─── (추가 명령어가 있으면 실행) ───
│
├── [실행 상태 리셋]
│   step_state_ 초기화
│   → 다음 execute() 호출 가능
│
└── 반환: Error::Ok
```

---

## 10. Runner에서 .PTE 필드가 적용되는 지점 총정리

| .PTE 필드 | 적용 시점 | Runner에서의 용도 |
|-----------|----------|-------------------|
| **execution_plan["forward"]** | `load_method("forward")` | 메인 추론 메서드 로드 |
| **execution_plan["get_max_seq_len"]** | `get_llm_metadata()` | `metadata["get_max_seq_len"]=2048` |
| **execution_plan["use_kv_cache"]** | `get_llm_metadata()` | TextPrefiller, TextTokenGenerator 생성 시 |
| **execution_plan["enable_dynamic_shape"]** | `get_llm_metadata()` | parallel prefill 여부 결정 |
| **execution_plan["get_eos_ids"]** | `get_eos_ids()` | 토큰 생성 루프 종료 조건 |
| **forward.values (Tensor, constant)** | `Method::init()` | 가중치 → mmap 포인터 직접 연결 |
| **forward.values (Tensor, mutable)** | `Method::init()` | KV 캐시, 활성화 → 계획된 메모리 할당 |
| **forward.values (Tensor, input)** | `Method::set_input()` | 토큰 ID, 캐시 위치 → 매 실행마다 설정 |
| **forward.inputs** | `Method::set_input()` | 어떤 values가 외부 입력인지 |
| **forward.outputs** | `Method::get_outputs()` | 어떤 values가 출력(logits)인지 |
| **forward.delegates[0].id** | `Method::init()` | "XnnpackBackend" 식별 |
| **forward.delegates[0].processed** | `Method::init()` | XNNPACK 바이너리 로드 & 초기화 |
| **forward.non_const_buffer_sizes** | `load_method()` | 활성화 메모리 50MB 할당 |
| **forward.chains[0].instructions** | `Method::execute()` | DelegateCall → XNNPACK 실행 |
| **constant_segment.offsets** | `parseTensor()` | 각 가중치의 세그먼트 내 위치 |
| **segments[0]** | `Program::load()` | 가중치 세그먼트 위치/크기 |
| **segments[1]** | `Method::init()` | XNNPACK delegate 바이너리 위치/크기 |
| **Tensor.sizes** | `parseTensor()` | 텐서 shape (MethodMeta로도 조회) |
| **Tensor.scalar_type** | `parseTensor()` | Float/Half/Int4 등 데이터 타입 |
| **Tensor.shape_dynamism** | `parseTensor()` | DYNAMIC_BOUND → 동적 shape 허용 |
| **Tensor.allocation_info** | `parseTensor()` | 메모리 풀 ID + 오프셋 |
| **MethodMeta.num_inputs()** | `TextDecoderRunner::step()` | KV 캐시 사용 여부 (>1이면 사용) |
| **MethodMeta.input_tensor_meta(1)** | `populate_start_pos_or_cache_position()` | cache_position 텐서 shape 결정 |

---

## 11. TikToken 토크나이저 상세 분석

### 11.1 아키텍처

```
pytorch/tokenizers 라이브러리
└── Tiktoken 클래스
    ├── load(path) → .model 파일 로드
    ├── encode(text, bos, eos) → vector<uint64_t>
    ├── decode(prev_token, cur_token) → string
    ├── vocab_size() → size_t
    ├── bos_tok() → uint64_t
    └── eos_tok() → uint64_t
```

### 11.2 LLaMA 3 TikToken 설정

**파일**: `examples/models/llama/tokenizer/llama_tiktoken.cpp`

```cpp
// LLaMA 3의 TikToken 생성
get_tiktoken_for_llama(Version::Default):
  special_tokens = [
    "<|begin_of_text|>",         // index 0 → BOS (token ID: 128000)
    "<|end_of_text|>",           // index 1 → EOS (token ID: 128001)
    "<|reserved_special_token_0|>",  // 128002
    "<|reserved_special_token_1|>",  // 128003
    "<|finetune_right_pad_id|>",     // 128004
    "<|step_id|>",                   // 128005
    "<|start_header_id|>",           // 128006
    "<|end_header_id|>",             // 128007
    "<|eom_id|>",                    // 128008
    "<|eot_id|>",                    // 128009
    "<|python_tag|>",                // 128010
    "<|reserved_special_token_2|>",  // 128011
    ... (총 256개까지 패딩)
  ]

  return Tiktoken(special_tokens, bos_index=0, eos_index=1)
  → bos_tok() = 128000 (special_tokens의 vocab 시작 ID + index 0)
  → eos_tok() = 128001 (special_tokens의 vocab 시작 ID + index 1)
```

### 11.3 토크나이저 파일 포맷 (.model)

TikToken의 `.model` 파일은 base64 인코딩된 BPE 병합 규칙 목록입니다:

```
tokenizer.model 파일 구조 (텍스트 파일):
  <base64_encoded_token> <rank>
  <base64_encoded_token> <rank>
  ...

예시 (처음 몇 줄):
  IQ== 0        ← " " (공백) → rank 0
  ICA= 1        ← "  " (공백 2개) → rank 1
  ICAG 2        ← "   " (공백 3개) → rank 2
  ...
  dGhl 1234     ← "the" → rank 1234
  ...

총 라인 수: ~128000 (일반 토큰)
+ 256개 특수 토큰 (코드에서 추가)
= vocab_size ~128256
```

### 11.4 load() 과정

```
tokenizer->load("tokenizer.model")
│
├── [1] 파일 읽기
│   각 줄: "<base64> <rank>" 파싱
│
├── [2] BPE 병합 테이블 구축
│   for each line:
│     token_bytes = base64_decode(base64_str)
│     rank = parse_int(rank_str)
│     encoder[token_bytes] = rank
│     decoder[rank] = token_bytes
│
├── [3] 특수 토큰 추가
│   base_vocab_size = encoder.size()  (예: 128000)
│   for (i = 0; i < special_tokens.size(); i++):
│     token_str = special_tokens[i]  (예: "<|begin_of_text|>")
│     token_id = base_vocab_size + i  (예: 128000 + 0 = 128000)
│     special_encoder[token_str] = token_id
│     special_decoder[token_id] = token_str
│
├── [4] BOS/EOS ID 설정
│   bos_id = base_vocab_size + bos_index = 128000 + 0 = 128000
│   eos_id = base_vocab_size + eos_index = 128000 + 1 = 128001
│
└── [5] 정규식 패턴 컴파일
    LLaMA 3 TikToken 패턴 (GPT-4 계열):
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|
      \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    → 영어 축약형, 단어, 숫자(3자리), 구두점, 공백 등을 분리
```

### 11.5 encode() 과정

```
tokenizer->encode("The answer to the ultimate question is", bos=1, eos=0)
│
├── [1] BOS 토큰 추가 (bos=1)
│   tokens = [128000]  ← <|begin_of_text|>
│
├── [2] 정규식으로 텍스트 분할
│   regex_split("The answer to the ultimate question is")
│   → ["The", " answer", " to", " the", " ultimate", " question", " is"]
│
├── [3] 각 조각에 BPE 인코딩 적용
│   "The" → byte_pair_encode():
│     (a) UTF-8 바이트로 변환: [84, 104, 101]
│     (b) 초기 토큰: [(84), (104), (101)]
│     (c) BPE 병합 반복:
│         - (104, 101) → "he" 병합 가능, rank=1234
│         - (84, 1234) → "The" 병합 가능, rank=791
│     (d) 최종: [791]
│
│   " answer" → [4320]     (공백+answer 한 토큰)
│   " to" → [311]
│   " the" → [279]
│   " ultimate" → [17139]
│   " question" → [3488]
│   " is" → [374]
│
├── [4] 결과 결합
│   tokens = [128000, 791, 4320, 311, 279, 17139, 3488, 374]
│
└── [5] EOS 토큰 (eos=0이므로 추가 안 함)

반환: [128000, 791, 4320, 311, 279, 17139, 3488, 374]
      (8개 토큰)
```

### 11.6 decode() 과정

```
tokenizer->decode(prev_token=791, cur_token=4320)
│
├── [1] cur_token이 특수 토큰인지 확인
│   4320 < 128000 → 일반 토큰
│
├── [2] decoder 테이블에서 바이트 시퀀스 조회
│   decoder[4320] → [32, 97, 110, 115, 119, 101, 114]
│                    → " answer" (UTF-8)
│
├── [3] 바이트를 문자열로 변환
│   → " answer"
│
└── 반환: " answer"

특수 토큰 디코딩 예시:
  decode(x, 128000) → "<|begin_of_text|>"
  decode(x, 128001) → "<|end_of_text|>"
  decode(x, 128009) → "<|eot_id|>"
```

### 11.7 Runner에서의 토크나이저 사용 흐름

```
[생성 시]
create_text_llm_runner()
  └── tokenizer->vocab_size() → 128256  ← metadata에 저장
      tokenizer->bos_tok() → 128000     ← metadata["get_bos_id"]
      tokenizer->eos_tok() → 128001     ← eos_ids 기본값

[Prefill 시]
TextLLMRunner::generate()
  └── tokenizer->encode(prompt, bos=0, eos=0)
      → [791, 4320, 311, 279, 17139, 3488, 374]
      (num_bos=0이므로 BOS 미추가)

[첫 토큰 출력]
  tokenizer->decode(cur_token, cur_token) → " 42" (예시)

[생성 루프]
TextTokenGenerator::generate()
  └── 매 반복:
      tokenizer->decode(prev_token, cur_token) → 텍스트 조각
      → token_callback(텍스트) → stdout 출력

[EOS 판정]
  cur_token ∈ eos_ids = {128001, 128009}?
  → 128001 (<|end_of_text|>) 또는 128009 (<|eot_id|>) 이면 생성 종료
```

### 11.8 LLaMA 2 vs LLaMA 3 토크나이저 차이

| 특성 | LLaMA 2 (SentencePiece) | LLaMA 3 (TikToken) |
|------|------------------------|---------------------|
| 라이브러리 | SentencePiece | TikToken (BPE) |
| 파일 형식 | `.model` (protobuf) | `.model` (base64 텍스트) |
| vocab_size | 32000 | 128000 (+256 특수) |
| BOS token | 1 (`<s>`) | 128000 (`<|begin_of_text|>`) |
| EOS token | 2 (`</s>`) | 128001 (`<|end_of_text|>`) |
| 특수 토큰 | 내장 | 코드에서 256개 추가 |
| 한글 지원 | 제한적 (바이트 폴백) | 양호 (BPE 직접 학습) |
| 토큰화 효율 | 영어 ~1.3 토큰/단어 | 영어 ~1.1 토큰/단어 |

```
load_tokenizer() 시도 순서:
  (1) Tekken (tekken.json 파일명일 때만)
  (2) HuggingFace JSON → tokenizer.json 파싱 시도
  (3) TikToken → tokenizer.model base64 파싱 시도  ← LLaMA 3
  (4) SentencePiece → tokenizer.model protobuf 파싱 시도  ← LLaMA 2
  (5) BPE (Llama2c) → tokenizer.bin 바이너리 파싱 시도

LLaMA 3의 경우:
  get_tiktoken_for_llama()로 특수 토큰 설정 후
  tiktoken->load("tokenizer.model") 호출
  → base64 BPE 테이블 로드 성공 → TikToken 사용 확정
```
