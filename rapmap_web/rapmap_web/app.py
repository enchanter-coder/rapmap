from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
from collections import deque
import heapq
import json
import os

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────
CSV_FILE = "DHH Database.csv"

# ── Load and build on startup ─────────────────────────────────────
df = None
G  = None

def load_data():
    global df
    data = pd.read_csv(CSV_FILE)
    data = data.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    data["featured_artists"] = data["featured_artists"].fillna("")

    def to_list(value):
        if pd.isna(value) or value == "":
            return []
        return [x.strip() for x in str(value).split(",")]

    data["language"] = data["language"].apply(to_list)
    data["vibe"]     = data["vibe"].apply(to_list)
    df = data
    print(f"  Loaded {len(df)} songs.")

def similarity_score(song1, song2):
    score = 0
    if set(song1["language"]) & set(song2["language"]):
        score += 3
    if set(song1["vibe"]) & set(song2["vibe"]):
        score += 3
    if song1["bpm_range"] == song2["bpm_range"]:
        score += 2
    if song1["city"] == song2["city"]:
        score += 2
    if song1["artist"] == song2["artist"]:
        score += 4
    if song1["artist"] in song2["featured_artists"]:
        score += 4
    if song2["artist"] in song1["featured_artists"]:
        score += 4
    return score

def build_graph():
    global G
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(
            int(row["song_id"]),
            title=row["title"],
            artist=row["artist"],
            language=row["language"],
            vibe=row["vibe"],
            bpm_range=row["bpm_range"],
            city=row["city"],
            featured_artists=row["featured_artists"]
        )
    for i, song1 in df.iterrows():
        for j, song2 in df.iterrows():
            if i >= j:
                continue
            score = similarity_score(song1, song2)
            if score >= 5:
                G.add_edge(int(song1["song_id"]), int(song2["song_id"]), weight=score)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# ── Algorithms ────────────────────────────────────────────────────

def bfs_recommend(seed_id, top_n=5):
    visited = set()
    queue   = deque()
    results = []
    queue.append((seed_id, 0))
    visited.add(seed_id)
    while queue and len(results) < top_n:
        current, hops = queue.popleft()
        neighbours = sorted(G[current].items(), key=lambda x: x[1]["weight"], reverse=True)
        for neighbour, data in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, hops + 1))
                if neighbour != seed_id:
                    node = G.nodes[neighbour]
                    results.append({
                        "song_id" : neighbour,
                        "title"   : node["title"],
                        "artist"  : node["artist"],
                        "score"   : data["weight"],
                        "hops"    : hops + 1,
                        "why"     : f"Similar language, vibe and scene — {hops+1} hop(s) from your song"
                    })
                if len(results) >= top_n:
                    break
    return results

def dfs_recommend(seed_id, top_n=5):
    visited = set()
    results = []
    def dfs(node, hops):
        if len(results) >= top_n:
            return
        visited.add(node)
        neighbours = sorted(G[node].items(), key=lambda x: x[1]["weight"])
        for neighbour, data in neighbours:
            if neighbour not in visited:
                if neighbour != seed_id:
                    node_data = G.nodes[neighbour]
                    results.append({
                        "song_id" : neighbour,
                        "title"   : node_data["title"],
                        "artist"  : node_data["artist"],
                        "score"   : data["weight"],
                        "hops"    : hops + 1,
                        "why"     : f"Deep discovery — {hops+1} hops into the graph"
                    })
                dfs(neighbour, hops + 1)
                if len(results) >= top_n:
                    return
    dfs(seed_id, 0)
    return results

def heuristic(node_id, target_vibe):
    node_vibes = G.nodes[node_id]["vibe"]
    return 0 if target_vibe in node_vibes else 5

def astar_recommend(seed_id, target_vibe, top_n=5):
    heap    = []
    visited = set()
    results = []
    heapq.heappush(heap, (0, seed_id, 0, 0))
    while heap and len(results) < top_n:
        cost, current, path_score, hops = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current != seed_id:
            node      = G.nodes[current]
            why_parts = []
            if target_vibe in node["vibe"]:
                why_parts.append(f"{target_vibe} vibe")
            if node["bpm_range"]:
                why_parts.append(f"{node['bpm_range']} BPM")
            if node["city"]:
                why_parts.append(f"{node['city']} scene")
            results.append({
                "song_id" : current,
                "title"   : node["title"],
                "artist"  : node["artist"],
                "score"   : path_score,
                "hops"    : hops,
                "why"     : " + ".join(why_parts) if why_parts else "similar features"
            })
        for neighbour, data in G[current].items():
            if neighbour not in visited:
                edge_weight = data["weight"]
                h           = heuristic(neighbour, target_vibe)
                new_cost    = cost + (10 - edge_weight) + h
                heapq.heappush(heap, (new_cost, neighbour, edge_weight, hops + 1))
    return results

# ── Graph data for visualization ──────────────────────────────────

def get_graph_data(seed_id=None):
    city_colors = {
        "Delhi"      : "#7F77DD",
        "Mumbai"     : "#D85A30",
        "Pune"       : "#639922",
        "Chandigarh" : "#BA7517",
        "Other"      : "#888780"
    }
    nodes = []
    for node_id in G.nodes():
        node  = G.nodes[node_id]
        color = city_colors.get(node["city"], "#888780")
        nodes.append({
            "id"     : node_id,
            "label"  : node["title"],
            "artist" : node["artist"],
            "city"   : node["city"],
            "color"  : "#1F3864" if node_id == seed_id else color,
            "size"   : 20 if node_id == seed_id else 10
        })
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({ "from": u, "to": v, "weight": data["weight"] })
    return { "nodes": nodes, "edges": edges }

# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    songs = df[["song_id", "title", "artist"]].to_dict("records")
    vibes = ["Aggressive", "Chill", "Conscious", "Motivational", "Sad"]
    top_connected = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    top_songs = []
    for song_id, degree in top_connected:
        row = df[df["song_id"] == song_id].iloc[0]
        top_songs.append({
            "title"       : row["title"],
            "artist"      : row["artist"],
            "connections" : degree
        })
    return render_template("index.html", songs=songs, vibes=vibes, top_songs=top_songs)

@app.route("/recommend", methods=["POST"])
def recommend():
    data        = request.json
    seed_id     = int(data["song_id"])
    target_vibe = data["vibe"]
    algorithm   = data["algorithm"]

    seed_node  = G.nodes[seed_id]
    seed_label = f"{seed_node['title']} — {seed_node['artist']}"

    if algorithm == "bfs":
        results = bfs_recommend(seed_id)
        algo_label = "BFS — Closest songs first"
        algo_desc  = "Explores level by level. Finds songs closest to your seed — safe, expected picks."
    elif algorithm == "dfs":
        results = dfs_recommend(seed_id)
        algo_label = "DFS — Deep exploration"
        algo_desc  = "Dives deep into the graph. Finds niche, unexpected, underground discoveries."
    elif algorithm == "astar":
        results = astar_recommend(seed_id, target_vibe)
        algo_label = f"A* — Smartest match to '{target_vibe}'"
        algo_desc  = "Uses your target vibe as a guide. Finds the best mood match intelligently."
    else:
        bfs    = bfs_recommend(seed_id)
        dfs    = dfs_recommend(seed_id)
        astar  = astar_recommend(seed_id, target_vibe)
        return jsonify({
            "mode"       : "compare",
            "seed_label" : seed_label,
            "target_vibe": target_vibe,
            "bfs"        : { "results": bfs,   "label": "BFS — Closest songs first",        "desc": "Safe, expected picks close to your seed." },
            "dfs"        : { "results": dfs,   "label": "DFS — Deep exploration",            "desc": "Niche, adventurous, unexpected discoveries." },
            "astar"      : { "results": astar, "label": f"A* — Best match to '{target_vibe}'", "desc": "Smartest match targeting your chosen vibe." }
        })

    return jsonify({
        "mode"       : "single",
        "seed_label" : seed_label,
        "target_vibe": target_vibe,
        "algo_label" : algo_label,
        "algo_desc"  : algo_desc,
        "results"    : results
    })

@app.route("/graph")
def graph():
    seed_id = request.args.get("seed_id", type=int)
    data    = get_graph_data(seed_id)
    songs   = df[["song_id", "title", "artist"]].to_dict("records")
    return render_template("graph.html", graph_data=json.dumps(data), songs=songs, seed_id=seed_id)

@app.route("/graph-data")
def graph_data():
    seed_id = request.args.get("seed_id", type=int)
    return jsonify(get_graph_data(seed_id))

# ── Start ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Starting RapMap...")
    load_data()
    build_graph()
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True)
