import numpy as np
import matplotlib.pyplot as plt

# וקטור לדוגמה
vector = np.array([5, 3, 8, 2, 7, 6, 1, 4, 9, 0])

# יצירת גרף עם שני subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# סאבפלוט ראשון: טקסט מתמטי ב-LATEX
axs[0].axis('off')  # מסירים את הצירים
text = (
    r"$\mathbf{Vector\ Representation}$" + "\n\n"
    r"$\text{Let } \vec{v} = [v_0, v_1, \dots, v_n] \text{ be a vector.}$" + "\n"
    r"$\text{We plot } v_i \text{ as a function of its index } i.$" + "\n"
    r"$\text{For example: } \vec{v} = [5, 3, 8, 2, 7, 6, 1, 4, 9, 0].$"
)
axs[0].text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)

# סאבפלוט שני: ערכי הווקטור כפונקציה של האינדקס
axs[1].plot(range(len(vector)), vector, marker='o', color='b', label=r"$v_i$")
axs[1].set_title("Vector Element Values", fontsize=14)
axs[1].set_xlabel("Index ($i$)", fontsize=12)
axs[1].set_ylabel("Value ($v_i$)", fontsize=12)
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
