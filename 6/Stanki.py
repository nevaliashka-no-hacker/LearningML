import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path

# 1
filename = "telemetriya_stankov_360.csv"
if not Path(filename).exists():
    try:
        from google.colab import files
        print("Выберите файл:", filename)
        files.upload()
    except ImportError:
        raise FileNotFoundError(
            f"Поместите {filename} в папку с notebook и запустите ячейку снова."
        )

if not Path(filename).exists():
    raise FileNotFoundError(f"Файл {filename} не найден после загрузки.")

df = pd.read_csv(filename)
print(df.head())
print()
print("Информация по датасету:")
print(df.info())
print()
print("Количество станков:", df.shape[0])
print("Пропуски по столбцам:\n", df.isna().sum().to_string())
print("Типы оборудования:", df["machine_type"].value_counts().to_dict())
print("Режимы:", df["mode"].value_counts().to_dict())

# 2
stats_cols = ["temperature_C", "current_A", "vibration_before_mm_s"]

stats = pd.DataFrame({
    "mean": df[stats_cols].mean(),
    "median": df[stats_cols].median(),
    "mode": df[stats_cols].mode().iloc[0],
    "var": df[stats_cols].var(),
    "std": df[stats_cols].std(),
    "min": df[stats_cols].min(),
    "max": df[stats_cols].max(),
    "q1": df[stats_cols].quantile(0.25),
    "q3": df[stats_cols].quantile(0.75),
})
print(stats)

# 3
mean_temp_by_type = df.groupby("machine_type")["temperature_C"].mean()
print("Оценка математического ожидания температуры по типам станков:")
print(mean_temp_by_type.to_string())

diff = mean_temp_by_type.max() - mean_temp_by_type.min()
print(f"\nРазница между типами: {diff:.2f} °C")

# 4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df["temperature_C"].hist(ax=axes[0], bins=30, color="steelblue", edgecolor="black")
axes[0].set_title("Гистограмма температуры")
axes[0].set_xlabel("temperature_C")
axes[0].set_ylabel("Частота")

df["current_A"].hist(ax=axes[1], bins=30, color="darkorange", edgecolor="black")
axes[1].set_title("Гистограмма тока")
axes[1].set_xlabel("current_A")
axes[1].set_ylabel("Частота")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
df.boxplot(column="vibration_before_mm_s", by="mode", grid=True)
plt.title("Boxplot вибрации по режимам работы")
plt.suptitle("")
plt.xlabel("Режим работы")
plt.ylabel("vibration_before_mm_s")
plt.tight_layout()
plt.show()

print("Асимметрия (skew):")
print(df[["temperature_C", "current_A"]].skew().to_string())

# 5
from scipy import stats

idle_temp = df.loc[df["mode"] == "idle", "temperature_C"]

t_stat, p_value = stats.ttest_1samp(idle_temp, popmean=30)

print("H0: средняя температура в режиме idle = 30 °C")
print("H1: средняя температура в режиме idle ≠ 30 °C")
print(f"alpha = 0.05")
print(f"n = {len(idle_temp)}")
print(f"Выборочное среднее = {idle_temp.mean():.4f} °C")
print(f"t-статистика = {t_stat:.4f}")
print(f"p-value = {p_value:.6f}")

if p_value < 0.05:
    print("p-value < 0.05 → нулевая гипотеза отвергается: средняя температура значимо отличается от 30 °C.")
else:
    print("p-value >= 0.05 → нет оснований отвергнуть нулевую гипотезу: средняя температура значимо не отличается от 30 °C.")

# 6
from scipy import stats

work_df = df[df["mode"] == "work"]

temp_milling = work_df.loc[work_df["machine_type"] == "milling", "temperature_C"]
temp_turning = work_df.loc[work_df["machine_type"] == "turning", "temperature_C"]

plt.figure(figsize=(8, 6))
plt.boxplot([temp_milling, temp_turning], labels=["milling", "turning"])
plt.title("Температура фрезерных и токарных станков в режиме work")
plt.xlabel("Тип станка")
plt.ylabel("temperature_C")
plt.grid(True)
plt.tight_layout()
plt.show()

u_stat, p_value = stats.mannwhitneyu(temp_milling, temp_turning, alternative="two-sided")

print("H0: распределения температур фрезерных и токарных станков в режиме work одинаковы")
print("H1: распределения температур различаются")
print(f"alpha = 0.05")
print(f"n_milling = {len(temp_milling)}, n_turning = {len(temp_turning)}")
print(f"Медиана milling = {temp_milling.median():.4f} °C")
print(f"Медиана turning = {temp_turning.median():.4f} °C")
print(f"U-статистика = {u_stat:.4f}")
print(f"p-value = {p_value:.6f}")

if p_value < 0.05:
    print("p-value < 0.05 → нулевая гипотеза отвергается: температуры фрезерных и токарных станков в режиме work значимо различаются.")
else:
    print("p-value >= 0.05 → нет оснований отвергнуть нулевую гипотезу: значимых различий температур не обнаружено.")

# 7
from scipy import stats

vib_before = df["vibration_before_mm_s"]
vib_after = df["vibration_after_mm_s"]
diff = vib_after - vib_before

plt.figure(figsize=(10, 5))
plt.hist(diff, bins=30, color="seagreen", edgecolor="black")
plt.title("Распределение разности вибрации (after − before)")
plt.xlabel("Разность вибрации, мм/с")
plt.ylabel("Частота")
plt.grid(True)
plt.tight_layout()
plt.show()

w_stat, p_value = stats.wilcoxon(vib_before, vib_after)

print("H0: медиана разности vibration_after − vibration_before = 0 (вибрация не изменилась)")
print("H1: медиана разности ≠ 0 (вибрация изменилась после ТО)")
print(f"alpha = 0.05")
print(f"n = {len(diff)}")
print(f"Медиана before = {vib_before.median():.4f} мм/с")
print(f"Медиана after  = {vib_after.median():.4f} мм/с")
print(f"Медиана разности = {diff.median():.4f} мм/с")
print(f"W-статистика = {w_stat:.4f}")
print(f"p-value = {p_value:.6f}")

if p_value < 0.05:
    print("p-value < 0.05 → нулевая гипотеза отвергается: уровень вибрации значимо изменился после технического обслуживания.")
else:
    print("p-value >= 0.05 → нет оснований отвергнуть нулевую гипотезу: значимых изменений вибрации после ТО не обнаружено.")

# 8
from scipy import stats

current = df["current_A"]
temperature = df["temperature_C"]

rho, p_value = stats.spearmanr(current, temperature)

plt.figure(figsize=(8, 6))
plt.scatter(current, temperature, alpha=0.5, s=15, color="teal")
plt.title("Зависимость температуры от потребляемого тока")
plt.xlabel("current_A")
plt.ylabel("temperature_C")
plt.grid(True)
plt.tight_layout()
plt.show()

print("H0: монотонной зависимости между током и температурой нет (ρ = 0)")
print("H1: монотонная зависимость присутствует (ρ ≠ 0)")
print(f"alpha = 0.05")
print(f"n = {len(df)}")
print(f"Коэффициент Спирмена ρ = {rho:.4f}")
print(f"p-value = {p_value:.6e}")

abs_rho = abs(rho)
if abs_rho < 0.2:
    strength = "очень слабая"
elif abs_rho < 0.4:
    strength = "слабая"
elif abs_rho < 0.6:
    strength = "умеренная"
elif abs_rho < 0.8:
    strength = "заметная"
else:
    strength = "сильная"

direction = "положительная (рост тока сопровождается ростом температуры)" if rho > 0 else "отрицательная (рост тока сопровождается снижением температуры)"

print(f"Направление связи: {direction}")
print(f"Сила связи: {strength} (|ρ| = {abs_rho:.4f})")

if p_value < 0.05:
    print("p-value < 0.05 → нулевая гипотеза отвергается: монотонная зависимость статистически значима.")
else:
    print("p-value >= 0.05 → нет оснований отвергнуть нулевую гипотезу: значимой монотонной зависимости не обнаружено.")

# 9
from scipy import stats

groups = {mode: df.loc[df["mode"] == mode, "temperature_C"] for mode in ["idle", "work", "high_load"]}

plt.figure(figsize=(10, 6))
plt.boxplot([groups["idle"], groups["work"], groups["high_load"]], labels=["idle", "work", "high_load"])
plt.title("Распределение температуры по режимам работы")
plt.xlabel("Режим работы")
plt.ylabel("temperature_C")
plt.grid(True)
plt.tight_layout()
plt.show()

h_stat, p_value = stats.kruskal(groups["idle"], groups["work"], groups["high_load"])

print("H0: распределения температуры в режимах idle, work и high_load одинаковы")
print("H1: хотя бы одно распределение отличается от остальных")
print(f"alpha = 0.05")
for mode, vals in groups.items():
    print(f"n_{mode} = {len(vals)}, медиана = {vals.median():.4f} °C, среднее = {vals.mean():.4f} °C")
print(f"H-статистика = {h_stat:.4f}")
print(f"p-value = {p_value:.6e}")

if p_value < 0.05:
    print("p-value < 0.05 → нулевая гипотеза отвергается: распределения температуры по режимам значимо различаются.")
else:
    print("p-value >= 0.05 → нет оснований отвергнуть нулевую гипотезу: значимых различий между режимами не обнаружено.")
