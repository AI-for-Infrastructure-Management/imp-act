import igraph as ig
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from geopy.geocoders import Photon

# List of 50 cities in Germany
cities = [
    "Berlin",
    "Hamburg",
    "Munich",
    "Cologne",
    "Frankfurt",
    "Stuttgart",
    "Dusseldorf",
    "Dortmund",
    "Bremen",
    "Leipzig",
    "Hanover",
    "Nuremberg",
    "Duisburg",
    "Bonn",
    "Karlsruhe",
    "Augsburg",
    "Wiesbaden",
    "Kiel",
    "Halle",
    "Freiburg",
    "Erfurt",
    "Mainz",
    "Rostock",
    "Saarbrucken",
    "Potsdam",
    "Heidelberg",
    "Darmstadt",
    "Magdeburg",
    "Bielefeld",
    "Lubeck",
    "Oberhausen",
    "Oldenburg",
    "Mannheim",
    "Gelsenkirchen",
    "Essen",
    "Leverkusen",
    "Hagen",
    "Hamm",
    "Krefeld",
    "Wuppertal",
    "Braunschweig",
    "Kassel",
    "Halle",
    "Lubeck",
    "Monchengladbach",
    "Wurzburg",
    "Regensburg",
    "Ingolstadt",
    "Ulm",
    "Paderborn",
    "Recklinghausen",
]

# Initialize Photon geolocator
geolocator = Photon(user_agent="measurements")

city_coordinates = {}  # Dictionary to store city coordinates

# Retrieve coordinates for each city
for city in cities:
    location = geolocator.geocode(city + ", Germany")
    if location:
        city_coordinates[city] = (location.latitude, location.longitude)

# Create an empty graph
G = ig.Graph()

# Add vertices (cities) to the graph with their coordinates
for city, coords in city_coordinates.items():
    G.add_vertex(city, pos=coords)  # Coordinates stored as a vertex attribute

# Calculate proximity (considering adjacency)
# based on distances between city coordinates

# Threshold distance in kilometers for adjacency
# (adjust as needed to control edge sparsity)
threshold_distance = 250

for city1, coord1 in city_coordinates.items():
    for city2, coord2 in city_coordinates.items():
        if city1 != city2:
            # Calculate the distance between cities using geodesic distance
            distance = geodesic(coord1, coord2).kilometers
            if distance < threshold_distance:
                G.add_edge(city1, city2)

# Draw the graph with custom layout based
# on latitude (North on top, South on bottom)
# Extract node positions (city coordinates)
node_positions = {v.index: v["pos"] for v in G.vs}

sorted_positions = sorted(node_positions.items(), key=lambda x: x[1][1], reverse=True)

# Create a figure and axis
plt.figure(figsize=(10, 8))

# Draw edges
for edge in G.get_edgelist():
    city1, city2 = edge
    y1, x1 = node_positions[edge[0]]
    y2, x2 = node_positions[edge[1]]
    plt.plot([x1, x2], [y1, y2], color="blue", alpha=0.7)

# Draw nodes
for city, (y, x) in node_positions.items():
    plt.scatter(x, y, color="red")
    plt.text(x, y, G.vs[city]["name"], fontsize=8, ha="left")

plt.title("(Dummy) Connections between 50 Cities in Germany")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid()
plt.tight_layout()
plt.show()

# save the graph
G.save("germany.graphml")

# load the graph
G = ig.Graph.Read_GraphML("germany.graphml")
