import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Define bounds
x_min, x_max = -2, 2
x = np.linspace(x_min, x_max, 1000)

# Truncated normal PDF (normalized)
normal_pdf_full = norm.pdf(x)
normal_pdf = normal_pdf_full / np.trapz(normal_pdf_full, x)

# Uniform PDF over same range
uniform_pdf = np.full_like(x, 1 / (x_max - x_min))

# Create plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot distributions with thicker lines and clearer colors
ax.plot(x, normal_pdf, linewidth=4, color="#FFD700")     # Golden yellow
ax.plot(x, uniform_pdf, linewidth=4, color="#00CED1")    # Bright turquoise

# Truncation bounds (vertical dashed lines)
ax.axvline(x_min, linestyle="--", color="white", linewidth=2)
ax.axvline(x_max, linestyle="--", color="white", linewidth=2)

# Aesthetic adjustments
ax.set_xlim(x_min - 0.5, x_max + 0.5)
ax.set_ylim(0, max(normal_pdf.max(), uniform_pdf.max()) * 1.2)

# No tick markers or labels
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(bottom=False, left=False)

# Hide top/right spines, keep axis lines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Save figure with transparent background
plt.tight_layout()
plt.savefig("docs/source/_static/sst_importance_sampling_logo.png", dpi=300, transparent=True)
plt.show()