import matplotlib.pyplot as plt
import matplotlib.patches as patches

road_color_mapping = {
    'motorway': '#4B0082',
    'trunk': '#800080',
    'primary': '#9400D3',
    'secondary': '#9932CC',
    'tertiary': '#BA55D3',
    'unclassified': '#DA70D6',
    'residential': '#EE82EE',
}

fig = plt.figure(figsize=(12, 3))

for i, (key, color) in enumerate(road_color_mapping.items()):
    rect = patches.Rectangle((i, 0), 1, 1, facecolor=color)
    plt.gca().add_patch(rect)
    plt.text(i+0.5, 0.5, key, color='white', weight='bold',
             horizontalalignment='center', verticalalignment='center')

plt.xlim(0, len(road_color_mapping))
plt.ylim(0, 1)
plt.axis('off')  # Turn off the box around the plot
plt.show()
