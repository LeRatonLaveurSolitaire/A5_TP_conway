import matplotlib.pyplot as plt
import numpy as np

bar_name = ["Original", "Gridsize v1", "Gridsize v2", "Int to char","Others to short"]
FPS = [1.72,241.23,296.63,397.67,483.35]


fig, ax = plt.subplots(figsize=(6, 4))

# Create color gradient based on time values
colors = plt.cm.RdYlGn_r(np.linspace(1, 0, len(FPS)))

bars = ax.bar(bar_name, FPS, color=colors)

old_height = 0

for bar in bars:
    height = bar.get_height()
    
    if height > 4 :
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}\n(+{height/old_height - 1:.1%})', 
            ha='center', va='bottom')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', 
            ha='center', va='bottom')
    old_height = height
# Customize the plot
ax.set_ylabel('FPS')
#ax.set_xlabel('Optimisation added')
ax.set_title('Performance with different optimisations')
plt.ylim(0, 550)
# Rotate x-axis labels for better readability
plt.xticks(rotation=20, ha='right')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('fps.png')
plt.show()
