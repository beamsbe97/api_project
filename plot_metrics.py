import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("audio_out/decoded/metrics.csv")

# Take the first (or only) row and drop non-metric columns
metrics = df.drop(columns=["path", "offset", "duration"]).iloc[0]

# Bar plot of all metrics
plt.figure(figsize=(8, 4))
metrics.plot(kind="bar")
plt.ylabel("Value")
plt.title("Codec reconstruction metrics")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()