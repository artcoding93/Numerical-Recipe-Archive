---
title: "[수치해석 02] 이분법(Bisection Method): 해가 여러 개일 때 이분법은 어떤 해를 찾아낼까? (다중 해 분리 기법)"
date: 2026-09-14
tags: ["수치해석", "Numerical Methods", "Python", "Bisection Method", "Grid Search", "다중해"]
---

# [수치해석 02] 해가 여러 개일 때 이분법은 어떤 해를 찾아낼까? (다중 해 분리 기법)

> **요약:** 사잇값 정리에 기반한 이분법은 구간 안에 해가 여러 개 존재할 때 맹점을 드러냅니다. 해가 여러 개 존재하는 시스템에서 원하는 해를 누락 없이 완벽히 찾아내기 위한 **격자 탐색(Grid Search) + 이분법 2단계 하이브리드 알고리즘**을 다룹니다.

---

## 1. 이분법이 빠지는 다중 해(Multiple Roots)의 함정

[1호 포스팅]에서 다루었듯, 이분법의 기본 출발 조건은 탐색 구간 양 끝값의 부호가 달라야 한다는 점입니다 ($f(a) \cdot f(b) < 0$). 

하지만 이 조건은 구간 내에 **"해(Root)가 홀수 개(1개, 3개, 5개...) 존재함"**을 의미할 뿐, 해가 몇 개 존재하는지는 가려내지 못합니다.

```text
       f(x)
        ▲         Root 1       Root 2       Root 3
   f(a) ┼───────────★────────────★────────────★─────────────
        │          / \          / \          /
 ───────┼─────────/───\────────/───\────────/──────────────► x
        │        /     \      /     \      /
   f(b) ┼───────/───────\────/───────\────/────────────────
        │

```

* **경우 1: 해가 홀수 개 존재할 때 ($f(a) \cdot f(b) < 0$)**
* 이분법을 실행하면 중간 탐색 과정에서 계산되는 부호 변화 패턴에 따라 **여러 해 중 단 하나로만 무작위 수렴**합니다. 엔지니어가 원하는 특정 해를 선택할 수 없습니다.


* **경우 2: 해가 짝수 개 존재할 때 ($f(a) \cdot f(b) > 0$)**
* 구간 내부에서 함수가 x축을 여러 번 교차하더라도 양 끝값 부호가 같아집니다. 그 결과 이분법은 구간 내에 해가 존재함에도 불구하고 **"해 없음"으로 잘못 결론 내리고 종료**됩니다.

---

## 2. 실전 문제: 구조물의 공진 주파수(Resonant Frequency) 탐색

건축 구조물, 교량, 혹은 진동 제어 시스템에서는 고유 진동수 방정식 $f(\omega) = 0$을 만족하는 **공진 주파수 $\omega$가 여러 개 존재**합니다.

$$\sin(\omega) - 0.1\omega = 0$$

* $\omega_1$: 1차 공진 (건물이 좌우로 전체적으로 흔들림)
* $\omega_2$: 2차 공진 (건물 중간이 꺾이며 변형 발생)
* $\omega_3$: 3차 공진 (고주파 진동 부품 파손)

엔지니어는 파괴적인 공진 현상을 피하기 위해 시스템의 모든 공진 주파수 위치를 누락 없이 전부 가려내야 합니다. 이분법 하나만 믿고 무작위로 도출된 단 하나의 해만 반영해서 설계했다간 심각한 구조적 참사로 이어질 수 있습니다.

---

## 3. 해결책: Grid Search(격자 탐색) + Bisection 2단계 전략

이 문제를 해결하는 수치해석적 표준 전략은 "전역 탐색으로 해의 범위를 가두고 $\rightarrow$ 국소 탐색으로 정밀 수렴시킨다"는 2단계 하이브리드 접근법입니다.

1. **1단계 (Grid Search):** 전체 탐색 범위를 일정한 간격의 미세 구간(Sub-intervals)으로 쪼갠 뒤, 부호 변화($f(x_i) \cdot f(x_{i+1}) < 0$)가 발생하는 해의 존재 구간들을 수집(Root Isolation)합니다.
2. **2단계 (Bisection):** 격리된 각 세부 구간에 대해 개별적으로 이분법을 적용하여 정밀한 해들을 모두 도출합니다.

---

## 4. 파이썬 구현 코드 (Google Colab)

```python
import numpy as np

# 대상 비선형 방정식 (여러 해를 가짐)
def target_func(x):
    return np.sin(x) - 0.1 * x

# 1단계: 전체 영역을 성글게 쪼개어 해가 포함된 서브 구간 격리
def find_root_intervals(func, start, end, num_subintervals=100):
    x_grid = np.linspace(start, end, num_subintervals)
    intervals = []
    
    for i in range(len(x_grid) - 1):
        a_i, b_i = x_grid[i], x_grid[i+1]
        if func(a_i) * func(b_i) < 0:
            intervals.append((a_i, b_i))
            
    return intervals

# 2단계: 격리된 각 구간에 이분법을 적용해 모든 해 정밀 도출
def solve_all_roots(func, start, end, tol=1e-6):
    intervals = find_root_intervals(func, start, end)
    roots = []
    
    for a, b in intervals:
        for _ in range(100):
            c = (a + b) / 2.0
            f_c = func(c)
            
            if abs(f_c) < tol or (b - a) / 2.0 < tol:
                roots.append(c)
                break
                
            if func(a) * f_c < 0:
                b = c
            else:
                a = c
                
    return roots

# 실행: 범위 [-10, 10] 내의 모든 해 탐색
start_val, end_val = -10.0, 10.0
all_roots = solve_all_roots(target_func, start_val, end_val)

print(f"[{start_val}, {end_val}] 구간 내 발견된 모든 해:")
for idx, r in enumerate(all_roots, 1):
    print(f"  Root {idx}: x = {r:.5f} (f(x) = {target_func(r):.2e})")

```

### 📌 실행 결과
[-10.0, 10.0] 구간 내 발견된 모든 해:
  Root 1: x = -8.42320 (f(x) = -6.10e-07)
  Root 2: x = -7.06817 (f(x) = 7.11e-07)
  Root 3: x = -2.85234 (f(x) = 5.28e-07)
  Root 4: x = 0.00000 (f(x) = 0.00e+00)
  Root 5: x = 2.85234 (f(x) = -5.28e-07)
  Root 6: x = 7.06817 (f(x) = -7.11e-07)
  Root 7: x = 8.42320 (f(x) = 6.10e-07)


격자 분할 기법을 덧붙임으로써 단일 이분법으로는 놓치기 쉬웠던 3개의 모든 해를 원인 분석과 함께 누락 없이 안전하게 검출해냈습니다.

---

## 5. 수치해석적 인사이트: 하이브리드(Hybrid) 설계 철학

수치해석 및 최신 AI 시뮬레이션 분야에서 모든 조건에 완벽히 대응하는 단 하나의 전지전능한 알고리즘은 존재하지 않습니다.

* **Bisection:** 탐색 안정성은 100%이나 다중 해 구분 능력 및 탐색 속도가 아쉬움.
* **Grid Search:** 전역 탐색 능력은 뛰어나나 정밀 피팅에 계산 비용이 큼.

이 둘을 결합한 **"Grid Search + Bisection"** 구조는 수치해석의 가장 대표적인 하이브리드 패턴입니다. 이 설계 철학은 향후 다룰 **Levenberg-Marquardt 알고리즘의 초깃값 보정**이나 **최신 AI 기반 물리 시뮬레이션(PINN)의 도메인 분할 기법**에서도 동일한 수학적 원리로 재등장하게 됩니다.
