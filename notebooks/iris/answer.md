## 공통: 로딩 + DataFrame 만들기 (A1 포함)

```python
import pandas as pd
from sklearn.datasets import load_iris

# 1) 로드
iris = load_iris(as_frame=True)

# 2) DataFrame 구성 + 컬럼명 정리
df = iris.frame.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    "target": "species_id",
})

# 3) species 라벨 추가
df["species"] = df["species_id"].map({i: n for i, n in enumerate(iris.target_names)})

# 4) species_id 제거
df = df.drop(columns=["species_id"])

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
```

---

# A. 로딩·기본 구조 파악 (정답 코드)

## A2) shape / head(3) / tail(3) / info

```python
print(df.shape)
display(df.head(3))
display(df.tail(3))
df.info()
```

## A3) 결측치 개수

```python
missing = df.isna().sum()
display(missing)
```

## A4) species 빈도표

```python
species_counts = df["species"].value_counts()
display(species_counts)
```

---

# B. 단변량 기술통계(Overall) (정답 코드)

## B5) describe + (해석 보조용) mean-median 차이, std 정렬

```python
desc = df[feature_cols].describe()
display(desc)

mean_median_abs = (df[feature_cols].mean() - df[feature_cols].median()).abs().sort_values(ascending=False)
std_sorted = df[feature_cols].std().sort_values(ascending=False)

display(mean_median_abs)
display(std_sorted)
```

## B6) var / std / range 표

```python
var_std_range = pd.DataFrame({
    "var": df[feature_cols].var(),
    "std": df[feature_cols].std(),
    "range": df[feature_cols].max() - df[feature_cols].min(),
})
display(var_std_range)
```

## B7) Q1/Q2/Q3 및 IQR

```python
q = df[feature_cols].quantile([0.25, 0.5, 0.75]).T
q.columns = ["Q1", "Q2(median)", "Q3"]
q["IQR"] = q["Q3"] - q["Q1"]
display(q)
```

## B8) skew / kurtosis

```python
skew_kurt = pd.DataFrame({
    "skew": df[feature_cols].skew(),
    "kurtosis": df[feature_cols].kurt(),  # Fisher: 정규분포면 0
})
display(skew_kurt)
```

---

# C. 그룹별 기술통계(GroupBy) (정답 코드)

## C9) species별 평균

```python
group_means = df.groupby("species")[feature_cols].mean()
display(group_means)
```

## C10) species별 median / std (동시에)

```python
group_median_std = df.groupby("species")[feature_cols].agg(["median", "std"])
display(group_median_std)
```

## C11) species별 petal_length min/max

```python
petal_len_minmax = df.groupby("species")["petal_length"].agg(["min", "max"])
display(petal_len_minmax)
```

## C12) species별 petal_length IQR

```python
petal_len_iqr = df.groupby("species")["petal_length"].apply(lambda s: s.quantile(0.75) - s.quantile(0.25))
display(petal_len_iqr.to_frame("IQR"))
```

---

# D. 조건 필터링·정렬·인덱싱 (정답 코드)

## D13) setosa만 필터링 후 sepal_length 내림차순 상위 5개

```python
top5_setosa = (
    df[df["species"] == "setosa"]
    .sort_values("sepal_length", ascending=False)
    .head(5)
)
display(top5_setosa)
```

## D14) “꽃잎이 큰” 샘플 조건 + 샘플 수 + species 분포

```python
big_petal = df[(df["petal_length"] >= 5.0) & (df["petal_width"] >= 1.8)]

count_big = len(big_petal)
dist_big = big_petal["species"].value_counts()

print("조건 만족 샘플 수:", count_big)
display(dist_big)
display(big_petal.head(10))
```

## D15) versicolor 중 sepal_width가 전체 상위 10% (quantile)

```python
q90 = df["sepal_width"].quantile(0.9)

versi_top10 = df[(df["species"] == "versicolor") & (df["sepal_width"] >= q90)]
display(versi_top10)
print("개수:", len(versi_top10), "| 전체 90% 기준값:", q90)
```

## D16) species별 petal_length 평균이 큰 순서로 species 정렬

```python
species_order = (
    df.groupby("species")["petal_length"]
    .mean()
    .sort_values(ascending=False)
)

display(species_order)
print("정렬된 species:", species_order.index.tolist())
```

---

# E. 상관·공분산·관계 요약 (정답 코드)

## E17) 상관행렬 + 절댓값 기준 상위 2개 상관쌍 찾기

```python
corr = df[feature_cols].corr()
display(corr)

# 상관행렬을 long 형태로 바꾸고(자기자신 제외), 중복쌍 제거 후 |corr| 상위 2개
corr_long = (
    corr.where(~np.eye(len(feature_cols), dtype=bool))  # 대각선 NaN
        .stack()
        .rename("corr")
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2"})
)

# (var1,var2)와 (var2,var1) 중복 제거: 정렬된 튜플 키로 drop_duplicates
corr_long["pair"] = corr_long.apply(lambda r: tuple(sorted([r["var1"], r["var2"]])), axis=1)
corr_top2 = (
    corr_long.assign(abs_corr=corr_long["corr"].abs())
             .drop_duplicates("pair")
             .sort_values("abs_corr", ascending=False)
             .head(2)[["var1", "var2", "corr", "abs_corr"]]
)

display(corr_top2)
```

## E18) species별 상관행렬

```python
corr_by_species = {sp: g[feature_cols].corr() for sp, g in df.groupby("species")}

for sp, c in corr_by_species.items():
    print(f"\n=== {sp} correlation ===")
    display(c)
```

## E19) 공분산 행렬(cov) + (참고용) 상관과 비교

```python
cov = df[feature_cols].cov()
display(cov)

# (선택) 표준화 후 cov를 보면 corr에 가까워지는 것을 확인 가능
z = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std(ddof=1)
cov_of_z = z.cov()
display(cov_of_z)
```

---

# F. 이상치(outlier)·분포 점검 (정답 코드)

## F20) IQR(1.5*IQR) 기준 컬럼별 이상치 개수

```python
outlier_counts = {}

for col in feature_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()

outlier_counts = pd.Series(outlier_counts).sort_values(ascending=False)
display(outlier_counts)
```

## F21) species별 sepal_width의 IQR 이상치 개수(“박스플롯 기준”과 동일)

```python
def iqr_outlier_count(s: pd.Series) -> int:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())

sepal_width_outliers_by_species = (
    df.groupby("species")["sepal_width"]
      .apply(iqr_outlier_count)
      .sort_values(ascending=False)
      .to_frame("iqr_outlier_count")
)

display(sepal_width_outliers_by_species)
```

## F22) Z-score 기준 |z|>3 이상치 개수 vs IQR 비교

```python
z = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std(ddof=1)

z_outliers = (z.abs() > 3).sum().sort_values(ascending=False)
display(z_outliers)

# IQR 결과도 같이 보고 싶으면:
iqr_outliers = outlier_counts  # F20에서 만든 것
compare = pd.DataFrame({"IQR_outliers": iqr_outliers, "Z>|3|_outliers": z_outliers})
display(compare)
```
아래는 이전 흐름 그대로 **I. 미니 프로젝트(29~30번) “정답 코드(작성 예시)”**입니다.
(앞에서 만든 `df`, `feature_cols`가 이미 있다고 가정합니다.)

---

# I. 미니 프로젝트(종합) 정답 코드

## 29) “한 장짜리 EDA 요약” 만들기

요구사항(품질/전체 기술통계/그룹 요약/상관 상위2/이상치 IQR)을 **한 번에** 정리합니다.

```python
import pandas as pd
import numpy as np

# ---------- (1) 데이터 품질 ----------
quality = pd.Series({
    "n_rows": len(df),
    "n_cols": df.shape[1],
    "missing_total": int(df.isna().sum().sum()),
    "duplicates_rows": int(df.duplicated().sum()),
    "species_nunique": int(df["species"].nunique()),
})
quality_df = quality.to_frame("value")

# ---------- (2) 전체 기술통계 요약(핵심 5개 지표) ----------
# count/mean/std/min/50%/max 중에서 핵심 5개로 예: mean, std, min, median, max
overall_5 = df[feature_cols].agg(["mean", "std", "min", "median", "max"]).T
overall_5 = overall_5.rename(columns={"median": "median(50%)"})

# ---------- (3) species별 평균/표준편차 ----------
group_mean_std = df.groupby("species")[feature_cols].agg(["mean", "std"])

# ---------- (4) 상관 상위 2쌍 ----------
corr = df[feature_cols].corr()

corr_long = (
    corr.where(~np.eye(len(feature_cols), dtype=bool))   # 대각선 제외
        .stack()
        .rename("corr")
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2"})
)

corr_long["pair"] = corr_long.apply(lambda r: tuple(sorted([r["var1"], r["var2"]])), axis=1)

corr_top2 = (
    corr_long.assign(abs_corr=corr_long["corr"].abs())
             .drop_duplicates("pair")
             .sort_values("abs_corr", ascending=False)
             .head(2)[["var1", "var2", "corr", "abs_corr"]]
)

# ---------- (5) IQR 이상치 요약 ----------
outlier_counts = {}
outlier_bounds = []

for col in feature_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_bounds.append([col, q1, q3, iqr, lower, upper])

outlier_counts_s = pd.Series(outlier_counts).sort_values(ascending=False).to_frame("iqr_outlier_count")
outlier_bounds_df = pd.DataFrame(outlier_bounds, columns=["feature", "Q1", "Q3", "IQR", "lower", "upper"]).set_index("feature")

# ---------- 출력(한 장 요약처럼) ----------
print("\n=== 29) EDA 요약: Data Quality ===")
display(quality_df)

print("\n=== 29) EDA 요약: Overall Descriptive (5 stats) ===")
display(overall_5)

print("\n=== 29) EDA 요약: Group (species) mean/std ===")
display(group_mean_std)

print("\n=== 29) EDA 요약: Corr Top-2 pairs ===")
display(corr_top2)

print("\n=== 29) EDA 요약: IQR outlier counts ===")
display(outlier_counts_s)

print("\n(참고) IQR 경계값 테이블 ===")
display(outlier_bounds_df)
```

---

## 30) 위 요약 기반 “다음 분석 단계 제안(5줄 이내)” 자동 생성 코드

“문장”도 코드로 나오게 하려면 아래처럼 작성하면 됩니다.

```python
# 단일 변수 분리력(그룹 평균 차이 / 그룹 내 변동 대비)을 간단히 점수화해 우선순위 추천
# (정교한 통계검정은 아니고, EDA 단계의 휴리스틱입니다)

group_means = df.groupby("species")[feature_cols].mean()
group_stds = df.groupby("species")[feature_cols].std()

# 클래스 평균 간 거리(최대-최소) / (평균 std) 로 간단 점수
separation_score = (group_means.max() - group_means.min()) / group_stds.mean()
separation_score = separation_score.sort_values(ascending=False)

top_features = separation_score.index.tolist()

suggestion_lines = [
    f"1) 분류 전처리로는 표준화(StandardScaler)를 기본 적용하고, 스케일 영향이 큰 공분산/거리 기반 모델에서 특히 유효합니다.",
    f"2) 기술통계 기반 분리력 점수 상 상위 변수는 {top_features[:2]}이며, 우선 이 변수 중심으로 베이스라인 모델을 구성합니다.",
    "3) IQR 기준 이상치가 많은 변수(있다면)는 로버스트 스케일링/윈저라이징을 검토하되, Iris는 대체로 영향이 제한적입니다.",
    "4) 모델은 로지스틱 회귀(멀티클래스), SVM(RBF), RandomForest를 베이스라인으로 비교하고, 교차검증으로 성능을 평가합니다.",
    "5) 상관이 매우 높은 변수쌍이 있다면 다중공선성 관점에서 제거/차원축소(PCA)도 후보로 두되, 성능과 해석성으로 결정합니다.",
]

print("\n=== 30) 다음 분석 단계 제안(5줄) ===")
for line in suggestion_lines:
    print(line)
```

아래는 Iris 데이터셋 기준으로 **시각화 연습 문제(V1~V10)**를 추가한 것입니다. 각 항목은 **문제 + 정답 코드**가 함께 있습니다.
전제: 앞에서 사용한 `df`, `feature_cols`가 이미 존재한다고 가정합니다. (없다면 맨 아래 “공통 준비 코드”를 먼저 실행하세요.)

---

## V1. species 분포 막대그래프

**문제:** `species`별 샘플 개수를 막대그래프로 그리세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

counts = df["species"].value_counts()

plt.figure()
plt.bar(counts.index, counts.values)
plt.title("Species Counts")
plt.xlabel("species")
plt.ylabel("count")
plt.show()
```

---

## V2. 연속형 4개 특성 히스토그램(2x2)

**문제:** `feature_cols` 4개에 대해 히스토그램을 2x2로 배치해 그리세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.ravel()

for ax, col in zip(axes, feature_cols):
    ax.hist(df[col], bins=15)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("count")

plt.tight_layout()
plt.show()
```

---

## V3. species별 박스플롯(특성 1개)

**문제:** `petal_length`를 `species`별 박스플롯으로 그리세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

species_order = sorted(df["species"].unique())
data = [df.loc[df["species"] == sp, "petal_length"] for sp in species_order]

plt.figure()
plt.boxplot(data, labels=species_order, showfliers=True)
plt.title("Petal Length by Species (Boxplot)")
plt.xlabel("species")
plt.ylabel("petal_length")
plt.show()
```

---

## V4. species별 박스플롯(특성 4개를 한 번에)

**문제:** 4개 연속형 특성 각각에 대해 `species`별 박스플롯을 2x2로 그리세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

species_order = sorted(df["species"].unique())

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.ravel()

for ax, col in zip(axes, feature_cols):
    data = [df.loc[df["species"] == sp, col] for sp in species_order]
    ax.boxplot(data, labels=species_order, showfliers=True)
    ax.set_title(f"{col} by Species")
    ax.set_xlabel("species")
    ax.set_ylabel(col)

plt.tight_layout()
plt.show()
```

---

## V5. species별 바이올린 플롯(특성 1개)

**문제:** `petal_width`를 `species`별 바이올린 플롯으로 표현하세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

species_order = sorted(df["species"].unique())
data = [df.loc[df["species"] == sp, "petal_width"] for sp in species_order]

plt.figure()
plt.violinplot(data, showmeans=True, showmedians=True)
plt.xticks(range(1, len(species_order) + 1), species_order)
plt.title("Petal Width by Species (Violin)")
plt.xlabel("species")
plt.ylabel("petal_width")
plt.show()
```

---

## V6. 산점도: sepal_length vs sepal_width (species로 색 구분)

**문제:** `sepal_length`-`sepal_width` 산점도를 그리고, `species`에 따라 색이 달라지도록 하세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

species_codes = df["species"].astype("category").cat.codes  # 0,1,2
species_names = df["species"].astype("category").cat.categories

plt.figure()
sc = plt.scatter(df["sepal_length"], df["sepal_width"], c=species_codes)
plt.title("Sepal Length vs Sepal Width (colored by species)")
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")

# 범례(legend) 수동 구성
handles, _ = sc.legend_elements()
plt.legend(handles, species_names, title="species")
plt.show()
```

---

## V7. 산점도: petal_length vs petal_width + 클래스별 평균점 표시

**문제:** `petal_length`-`petal_width` 산점도를 그리고, 각 species의 평균점(centroid)을 X로 표시하세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

species_codes = df["species"].astype("category").cat.codes
species_names = df["species"].astype("category").cat.categories

plt.figure()
sc = plt.scatter(df["petal_length"], df["petal_width"], c=species_codes)
plt.title("Petal Length vs Petal Width + Centroids")
plt.xlabel("petal_length")
plt.ylabel("petal_width")

centroids = df.groupby("species")[["petal_length", "petal_width"]].mean()
plt.scatter(centroids["petal_length"], centroids["petal_width"], marker="x", s=100)

handles, _ = sc.legend_elements()
plt.legend(handles, species_names, title="species")
plt.show()
```

---

## V8. 산점도 행렬(Scatter Matrix)

**문제:** 4개 연속형 특성에 대해 산점도 행렬(Scatter Matrix)을 그리세요. (대각선은 히스토그램)

**정답 코드:**

```python
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

axes = scatter_matrix(df[feature_cols], figsize=(10, 10), diagonal="hist")
plt.suptitle("Scatter Matrix (features)", y=1.02)
plt.show()
```

---

## V9. 상관행렬 히트맵(Heatmap) — matplotlib만 사용

**문제:** `feature_cols`의 상관행렬을 히트맵으로 시각화하고, 축에 컬럼명을 표시하세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt
import numpy as np

corr = df[feature_cols].corr().values

plt.figure(figsize=(6, 5))
plt.imshow(corr, aspect="auto")
plt.title("Correlation Heatmap (features)")
plt.colorbar()

plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
plt.yticks(range(len(feature_cols)), feature_cols)

# 셀 값 텍스트(선택)
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

plt.tight_layout()
plt.show()
```

---

## V10. species별 평균±표준편차 에러바(예: petal_length)

**문제:** `species`별 `petal_length`의 평균과 표준편차를 막대(또는 점)+에러바로 표현하세요.

**정답 코드:**

```python
import matplotlib.pyplot as plt

stats = df.groupby("species")["petal_length"].agg(["mean", "std"])
x = range(len(stats.index))

plt.figure()
plt.errorbar(x, stats["mean"], yerr=stats["std"], fmt="o", capsize=5)
plt.title("Petal Length: Mean ± Std by Species")
plt.xticks(x, stats.index)
plt.xlabel("species")
plt.ylabel("petal_length")
plt.show()
```

---

# 공통 준비 코드(필요한 경우에만)

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={
    "sepal length (cm)":"sepal_length",
    "sepal width (cm)":"sepal_width",
    "petal length (cm)":"petal_length",
    "petal width (cm)":"petal_width",
    "target":"species_id"
})
df["species"] = df["species_id"].map({i:n for i,n in enumerate(iris.target_names)})
df = df.drop(columns=["species_id"])
feature_cols = ["sepal_length","sepal_width","petal_length","petal_width"]
```

원하시면, 위 시각화 문제를 **난이도 순으로 재배열**하거나, “정답 코드”를 **함수형(예: `plot_corr(df, cols)` 형태)**으로 리팩터링해 드릴 수 있습니다.
