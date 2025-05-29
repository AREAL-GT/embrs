import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from collections import defaultdict
import requests
import simplekml

# Assume fetch_osm_roads is already defined above and imported
# (I will redefine it here quickly for completeness if you want)

def fetch_osm_roads(bounds: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    left, bottom, right, top = bounds
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    (way["highway"]
    ({bottom}, {left}, {top}, {right});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"Querying OSM for road data in bounds: {bounds}")
    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code != 200:
        print(f"WARNING: Request failed with status {response.status_code}")
        return []

    osm_data = response.json()

    nodes = {elem['id']: (elem['lon'], elem['lat']) for elem in osm_data['elements'] if elem['type'] == 'node'}

    roads = []
    for elem in osm_data['elements']:
        if elem['type'] == 'way' and 'highway' in elem['tags']:
            tags = elem['tags']
            road_type = tags.get('highway')
            road_coords = [nodes[node_id] for node_id in elem['nodes'] if node_id in nodes]
            if not road_coords:
                continue

            road_info = {
                'coords': road_coords,
                'highway': road_type,
                'width': tags.get('width'),
                'est_width': tags.get('est_width'),
                'lanes': tags.get('lanes'),
                'lane_width': tags.get('lane_width'),
                'shoulder_width': tags.get('shoulder:width'),
                'surface': tags.get('surface'),
            }

            roads.append(road_info)

    print(f"Fetched {len(roads)} roads from OSM.")
    return roads

def plot_roads(roads: List[Dict[str, Any]]):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    for road in roads:
        coords = road['coords']
        highway_type = road['highway']

        xs, ys = zip(*coords)

        # Try to determine width
        width_m = None
        if road['width']:
            try:
                width_m = float(road['width'])
            except ValueError:
                pass
        elif road['est_width']:
            try:
                width_m = float(road['est_width'])
            except ValueError:
                pass

        # Fallback plotting width if no width available
        if width_m:
            linewidth = width_m / 2.0  # Scale for visibility
        else:
            linewidth = 1.0

        # Plot the road
        ax.plot(xs, ys, label=highway_type, linewidth=linewidth, alpha=0.7)

        # Label the center of the road with its type
        mid_idx = len(xs) // 2
        ax.text(xs[mid_idx], ys[mid_idx], highway_type, fontsize=6, rotation=0)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('OSM Roads with Classification and Widths')
    plt.grid(True)
    plt.legend(fontsize=8, loc='upper right')
    plt.show()

def save_roads_to_kml(roads: List[Dict[str, Any]], filename: str):
    """
    Saves the list of roads into a KML file for viewing in Google Earth,
    grouping roads by highway type into folders.

    Args:
        roads (List[Dict]): List of road data.
        filename (str): Output KML filename.
    """
    kml = simplekml.Kml()

    # Group roads by highway type
    highway_groups = defaultdict(list)
    for road in roads:
        highway_type = road['highway']
        highway_groups[highway_type].append(road)

    for highway_type, roads_list in highway_groups.items():
        folder = kml.newfolder(name=highway_type)

        for road in roads_list:
            coords = road['coords']
            width = road.get('width') or road.get('est_width')
            lanes = road.get('lanes')
            surface = road.get('surface')

            linestring = folder.newlinestring(
                name=highway_type,
                coords=coords  # (lon, lat) pairs
            )

            # Style
            linestring.style.linestyle.color = simplekml.Color.blue
            linestring.style.linestyle.width = 2

            # Emphasize major roads
            if highway_type in ['motorway', 'trunk', 'primary']:
                linestring.style.linestyle.width = 4
                linestring.style.linestyle.color = simplekml.Color.red

            # Optional description popup
            desc = f"Type: {highway_type}\n"
            if width:
                desc += f"Width: {width} m\n"
            if lanes:
                desc += f"Lanes: {lanes}\n"
            if surface:
                desc += f"Surface: {surface}\n"
            
            linestring.description = desc

    kml.save(filename)
    print(f"KML file saved as {filename}")


def main():
    # Pick a test bounding box
    # Example: El Dorado Hills, CA (wildfire interface zone)
    bounds = (-121.44, 46.6, -121.34, 46.7)  # (left, bottom, right, top)

    roads = fetch_osm_roads(bounds)
    if not roads:
        print("No roads fetched.")
        return

    plot_roads(roads)

    save_roads_to_kml(roads, "fetched_roads.kml")

if __name__ == "__main__":
    main()
