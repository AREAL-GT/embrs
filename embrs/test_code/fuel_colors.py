import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

fuel_names = {
    1: "Short grass", 2: "Timber grass", 3: "Tall grass", 4: "Chaparral",
    5: "Brush", 6: "Hardwood slash", 7: "Southern rough", 8: "Closed timber litter",
    9: "Hardwood litter", 10: "Timber litter", 11: "Light logging slash",
    12: "Medium logging slash", 13: "Heavy logging slash", 91: 'Urban', 92: 'Snow/ice',
    93: 'Agriculture', 98: 'Water', 99: 'Barren'
}

fuel_color_mapping = {
    1: 'xkcd:pale green', 2:'xkcd:lime', 3: 'xkcd:bright green', 4: 'xkcd:teal',
    5: 'xkcd:bluish green', 6: 'xkcd:greenish teal', 7: 'xkcd:light blue green',
    8: 'xkcd:pale olive', 9: 'xkcd:olive', 10: 'xkcd:light forest green',
    11: 'xkcd:bright olive', 12: 'xkcd:tree green', 13: 'xkcd:avocado green',
    91: 'xkcd:ugly purple', 92: 'xkcd:pale cyan', 93: "xkcd:perrywinkle",
    98: 'xkcd:water blue', 99: 'xkcd:black'
}

def hexagon(x_center, y_center, size):
    """Generate the coordinates for a hexagon."""
    return [(x_center + size * sin(theta), y_center + size * cos(theta))
            for theta in [i * (2 * pi / 6) for i in range(6)]]


# Initialize the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')

from math import sin, cos, pi

radius = 1.0
vertical_spacing = 2.5
horizontal_spacing = 4 * radius
text_width = 12

# Starting coordinates
x, y = 0, 0 

for idx, key in enumerate(fuel_names):
    hex_coords = hexagon(x, y, radius)
    hex_patch = patches.Polygon(hex_coords, facecolor=fuel_color_mapping[key], edgecolor='black')
    ax.add_patch(hex_patch)
    
    # Add wrapped label
    wrapped_text = textwrap.fill(fuel_names[key], text_width)
    ax.text(x, y - 2 * radius, wrapped_text, ha='center', va='center', fontsize=8)

    # Update x, y for next hexagon
    if (idx + 1) % 6 == 0:  # Go to the next row after every 6 hexagons
        x = 0
        y -= vertical_spacing * 2  # double vertical spacing to move to the next row
    else:
        x += horizontal_spacing

ax.autoscale_view()
plt.show()