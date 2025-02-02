# יש לאתחל מחדש את כל הקוד לאחר איפוס הסביבה

import os
import numpy as np
import matplotlib.pyplot as plt

# יצירת תיקייה לשמירת התמונות
output_dir = r"C:\Users\ASUS\Documents\code_images\overleaf_images\presentation\final_presentation/reflection_frames1"
os.makedirs(output_dir, exist_ok=True)


# פונקציה לרפלקציה של נקודה ביחס לקו
def reflect(point, line_point, line_normal):
    """ מבצע רפלקציה של נקודה סביב קו נתון """
    line_normal = line_normal / np.linalg.norm(line_normal)  # מנרמל את הווקטור הנורמלי
    diff = point - line_point
    projection = np.dot(diff, line_normal) * line_normal
    return 2 * projection - point


# הגדרת שני קווים בזוויות חדות יותר
line1_point = np.array([0, 0])
line1_normal = np.array([1, -3])  # קו תלול במקום y = x
line2_point = np.array([0, 0])
line2_normal = np.array([1, 3])  # קו תלול במקום y = -x

# נקודת התחלה רנדומלית
np.random.seed(42)
start_point = np.random.uniform(-1, 1, size=2)

# מספר האיטרציות
iterations = 100
trajectory = [start_point]

point = start_point
for _ in range(iterations):
    point = reflect(point, line1_point, line1_normal)  # רפלקציה סביב הקו הראשון
    trajectory.append(point)
    point = reflect(point, line2_point, line2_normal)  # רפלקציה סביב הקו השני
    trajectory.append(point)

trajectory = np.array(trajectory)

# שמירת כל פריים כתמונה
image_paths = []
for i in range(1, iterations + 1):  # רק 50 תמונות, לא כולל את כל הטרייל
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axline(line1_point, slope=3, color='black', linestyle="--", label="Steep Line 1")
    ax.axline(line2_point, slope=-3, color='black', linestyle="--", label="Steep Line 2")

    current_trajectory = trajectory[:i + 1]  # רק עד האיטרציה הנוכחית
    ax.plot(current_trajectory[:, 0], current_trajectory[:, 1], 'o-', markersize=2, alpha=0.6, label="Trajectory")
    ax.scatter(start_point[0], start_point[1], color='red', label="Start Point", zorder=3)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f"Frame {i}")

    image_path = os.path.join(output_dir, f"frame_{i:03d}.png")
    plt.savefig(image_path)
    plt.close(fig)

    image_paths.append(image_path)

# הצגת הנתיב שבו נשמרו התמונות
output_dir
