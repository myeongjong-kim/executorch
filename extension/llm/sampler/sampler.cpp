/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This is a modified version of https://github.com/karpathy/llama2.c.git
// @lint-ignore-every LICENSELINT
/**
 * MIT License
 *
 * Copyright (c) 2023 Andrej
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <executorch/extension/llm/sampler/sampler.h>
#include <algorithm>
#include <ctime>

namespace executorch {
namespace extension {
namespace llm {

// ============================================================================
// [한글 주석] Sampler — 토큰 샘플링 알고리즘
// ============================================================================
// LLM의 logits(각 토큰에 대한 점수)를 확률 분포로 변환하고,
// 그 분포에서 다음 토큰을 선택하는 클래스
//
// 3가지 샘플링 전략:
//   (1) sample_argmax: temperature=0일 때, 가장 높은 확률의 토큰 선택 (결정적)
//   (2) sample_mult:   확률 분포에서 다항 샘플링 (기본)
//   (3) sample_topp:   top-p (nucleus) 샘플링 — 누적 확률 p까지의 토큰만 후보
//
// [예시 가정값]
//   vocab_size = 32000
//   temperature = 0.8 → inv_temperature = 1.25
//   topp = 0.9

// ── Greedy Argmax 샘플링 ──
// 가장 높은 확률을 가진 토큰의 인덱스를 반환
// temperature=0일 때 사용 (완전히 결정적)
//
// [예시]
//   probabilities = [..., 0.15(idx=29871), ..., 0.12(idx=29906), ...]
//   → max_p = 0.15, max_i = 29871 반환
template <typename T>
int32_t Sampler::sample_argmax(T* probabilities) {
  int max_i = 0;
  T max_p = probabilities[0];
  for (int i = 1; i < vocab_size_; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

// ── 다항(Multinomial) 샘플링 ──
// 확률 분포(합=1)에서 랜덤하게 토큰을 선택
// CDF(누적분포함수)를 순회하며 coin 값 위치의 토큰 반환
//
// [예시]
//   probabilities = [0.05, 0.10, 0.15, 0.12, ...]  (합=1)
//   coin = 0.23 (0~1 사이 난수)
//   CDF: 0.05 → 0.15 → 0.30 (> 0.23!) → index 2 반환
template <typename T>
int32_t Sampler::sample_mult(T* probabilities, float coin) {
  T cdf = 0.0;
  for (int i = 0; i < vocab_size_; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return vocab_size_ - 1; // 반올림 오차 대비
}

// ── Top-p (Nucleus) 샘플링 ──
// 누적 확률이 topp(0.9)를 초과하는 최소 집합에서만 샘플링
// 매우 낮은 확률의 토큰이 선택되는 것을 방지하여 품질 향상
//
// [예시] vocab_size=32000, topp=0.9
//   확률 분포 (softmax 후):
//     token 29871: 0.15 (" 42")
//     token 29906: 0.12 (" 4")
//     token 29946: 0.08 (" 8")
//     ... (수천 개의 낮은 확률 토큰)
//
//   [단계 1] cutoff = (1-0.9)/(32000-1) = 0.0000031
//            cutoff 이상인 토큰만 후보로 선별 (대부분 포함)
//
//   [단계 2] 확률 내림차순 정렬:
//            [(0.15, 29871), (0.12, 29906), (0.08, 29946), ...]
//
//   [단계 3] 누적 확률이 0.9 초과하는 지점까지만 유지:
//            0.15 → 0.27 → 0.35 → ... → 0.91 (초과! → last_idx)
//            나머지 토큰은 제외됨
//
//   [단계 4] 잘린 분포 내에서 coin으로 샘플링:
//            r = coin * cumulative_prob
//            CDF 순회하여 r 위치의 토큰 선택
template <typename T>
int32_t Sampler::sample_topp(T* probabilities, float coin) {
  int n = vocab_size_;
  int n0 = 0;

  // cutoff 이상인 토큰만 후보로 추출 (정렬 전 사전 필터링)
  std::unique_ptr<ProbIndex<T>[]> probindex =
      std::make_unique<ProbIndex<T>[]>(vocab_size_);

  const float cutoff = (1.0f - topp_) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }

  // 확률 내림차순 정렬
  auto compare = [](const ProbIndex<T>& a, const ProbIndex<T>& b) {
    return a.prob > b.prob;
  };
  std::sort(probindex.get(), probindex.get() + n0, compare);

  // 누적 확률이 topp(0.9)를 초과하는 지점에서 절단
  T cumulative_prob = 0;
  int last_idx = n0 - 1;
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp_) {
      last_idx = i;
      break;
    }
  }

  // 절단된 분포에서 샘플링
  const T& r = coin * cumulative_prob;
  T cdf = 0;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // 반올림 오차 대비
}

// ── Sampler 생성자 ──
// inv_temperature_: temperature의 역수 (0이면 greedy 모드)
// 예시: temperature=0.8 → inv_temperature_ = 1/0.8 = 1.25
//       temperature=0.0 → inv_temperature_ = 0 (greedy argmax)
Sampler::Sampler(
    int vocab_size,
    float temperature,
    float topp,
    unsigned long long rng_seed)
    : vocab_size_(vocab_size),
      inv_temperature_(static_cast<bool>(temperature) ? 1.0f / temperature : 0),
      topp_(topp),
      rng_state_(rng_seed) {}

// 간편 생성자: topp 기본값 0.9, rng_seed는 현재 시각
Sampler::Sampler(int vocab_size, float temperature)
    : vocab_size_(vocab_size),
      inv_temperature_(static_cast<bool>(temperature) ? 1.0f / temperature : 0),
      topp_(kTopp),          // 0.9
      rng_state_(std::time(nullptr)) {}

// ── Softmax 함수 ──
// logits(임의의 실수)를 확률 분포(합=1, 각 값 0~1)로 변환
// 수치 안정성을 위해 max값을 빼고 exp 계산
//
// [예시] vocab_size=32000
//   입력 logits: [2.5, 1.8, 3.1, 0.5, ...]  (32000개)
//   max_val = 3.1
//   exp 계산: [exp(2.5-3.1), exp(1.8-3.1), exp(0), exp(0.5-3.1), ...]
//           = [0.549, 0.273, 1.0, 0.074, ...]
//   sum = 0.549 + 0.273 + 1.0 + 0.074 + ... (모두 합산)
//   정규화: 각 값 / sum → 확률 분포 [0.05, 0.025, 0.09, 0.007, ...]
template <typename T>
static void softmax(T* x, int size) {
  T max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  T sum = 0;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

// ── XorShift 난수 생성기 ──
// 빠르고 가벼운 유사 난수 생성 (암호학적으로는 안전하지 않음)
// 샘플링에서 엔트로피 소스로 사용
static unsigned int random_u32(unsigned long long* state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// [0, 1) 범위의 float 난수 생성
static float random_f32(unsigned long long* state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

// ============================================================================
// [한글 주석] Sampler::sample() — 핵심 샘플링 로직
// ============================================================================
// logits 배열(shape [vocab_size])에서 다음 토큰을 선택
//
// [예시] vocab_size=32000, temperature=0.8 (inv_temperature=1.25)
//
//   [경우 1] temperature=0 (inv_temperature_=0):
//     → sample_argmax(logits) → 최고 logit의 인덱스 반환 (결정적)
//
//   [경우 2] temperature=0.8 (inv_temperature_=1.25):
//     (a) 온도 스케일링: logits[i] *= 1.25
//         → logits의 차이가 증폭됨 → 분포가 더 뾰족해짐
//         → 높은 확률 토큰이 더 자주 선택됨
//         (참고: temperature > 1이면 분포가 평탄해져 더 랜덤)
//
//     (b) softmax(logits, 32000) → 확률 분포로 변환
//         예시 (상위 5개):
//           token 29871 (" 42"):  0.15
//           token 29906 (" 4"):   0.12
//           token 29946 (" 8"):   0.08
//           ... (나머지 합산 = 0.65)
//
//     (c) coin = random_f32() → [0, 1) 사이 난수
//         예: coin = 0.23
//
//     (d) topp=0.9이므로 top-p 샘플링:
//         → sample_topp(probabilities, coin=0.23)
//         → 누적 확률 0.9까지의 토큰에서만 샘플링
//         → 반환: 29871 (" 42")
template <typename T>
int32_t Sampler::sample(T* logits) {
  int next;
  if (inv_temperature_ == 0.0f) {
    // Greedy: 가장 높은 logit의 인덱스 반환
    next = sample_argmax(logits);
  } else {
    // 온도 스케일링: logits *= inv_temperature (= 1/temperature)
    // temperature=0.8이면 logits *= 1.25 (차이 증폭)
    for (int q = 0; q < vocab_size_; q++) {
      logits[q] *= inv_temperature_;
    }
    // softmax 적용: logits → 확률 분포
    softmax(logits, vocab_size_);
    // 난수 생성 (샘플링의 엔트로피 소스)
    float coin = random_f32(&rng_state_);
    // topp에 따라 샘플링 방식 선택
    if (topp_ <= 0 || topp_ >= 1) {
      // topp가 범위 밖이면 단순 다항 샘플링
      next = sample_mult(logits, coin);
    } else {
      // top-p (nucleus) 샘플링: 누적 확률 topp까지의 토큰에서만 선택
      next = sample_topp(logits, coin);
    }
  }
  return next;
}

template int32_t Sampler::sample<float>(float* logits);
template int32_t Sampler::sample<uint16_t>(uint16_t* logits);
template int32_t Sampler::sample<executorch::aten::Half>(
    executorch::aten::Half* logits);
template int32_t Sampler::sample<executorch::aten::BFloat16>(
    executorch::aten::BFloat16* logits);

} // namespace llm
} // namespace extension
} // namespace executorch
