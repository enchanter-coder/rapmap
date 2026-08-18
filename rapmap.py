import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # It disables GUI and saves graph as image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
import heapq

'''
 RAPMAP: Desi Hip Hop Recommender
 Uses BFS, DFS, and A*
 Dataset: DHH Database.csv

'''

CSV_FILE = r"DHH Database.csv"

VIBES = ["Aggressive", "Chill", "Conscious", "Motivational", "Sad"]


# Step 1: Load dataset and clean basic columns


def load_data(filepath):
    print("  Loading dataset...")
    df = pd.read_csv(filepath)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    df["featured_artists"] = df["featured_artists"].fillna("")

    def to_list(value):
        if pd.isna(value) or value == "":
            return []
        return [x.strip() for x in str(value).split(",")]

    df["language"] = df["language"].apply(to_list)
    df["vibe"]     = df["vibe"].apply(to_list)
    print(f"  {len(df)} songs loaded successfully.")
    return df



#  STEP 2 — SIMILARITY FUNCTION


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



#  STEP 3 — Make Graph


def build_graph(df, threshold=5):
    print("  Building similarity graph...")
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(
            row["song_id"],
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
            if score >= threshold:
                G.add_edge(song1["song_id"], song2["song_id"], weight=score)

    print(f"  Graph ready — {G.number_of_nodes()} songs, {G.number_of_edges()} connections.")
    return G



#  HELPERS
'''
To make the code easier to read and prevent repetition, I made helper functions. 
Node attributes are retrieved from the graph by get_song and formatted into a user-friendly output by song_label.
'''

def get_song(G, song_id):
    return G.nodes[song_id]

def song_label(G, song_id):
    s = get_song(G, song_id)
    return f"{s['title']} — {s['artist']}"



#  STEP 4 —  MENU

def show_song_list(df):
    print("\n  " + "-" * 60)
    print("  ALL SONGS IN RAPMAP DATABASE")
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        print(f"  {str(row['song_id']):<4} {row['title']:<30} — {row['artist']}")
    print("  " + "-" * 60)

def pick_song(df):
    show_song_list(df)
    while True:
        try:
            choice = int(input("\n  Enter the number of your seed song: "))
            match  = df[df["song_id"] == choice]
            if not match.empty:
                title  = match["title"].values[0]
                artist = match["artist"].values[0]
                print(f"\n  You selected: {title} — {artist}")
                return choice
            else:
                print(f"  Invalid. Enter a number between 1 and {len(df)}.")
        except ValueError:
            print("  Please enter a valid number.")

def pick_vibe():
    print("\n  " + "-" * 60)
    print("  SELECT TARGET VIBE")
    print("  " + "-" * 60)
    for i, vibe in enumerate(VIBES, 1):
        print(f"  {i}. {vibe}")
    print("  " + "-" * 60)
    while True:
        try:
            choice = int(input("\n  Enter vibe number: "))
            if 1 <= choice <= len(VIBES):
                selected = VIBES[choice - 1]
                print(f"  Target vibe: {selected}")
                return selected
            else:
                print(f"  Enter a number between 1 and {len(VIBES)}.")
        except ValueError:
            print("  Please enter a valid number.")

def pick_algorithm():
    print("\n  " + "-" * 60)
    print("  SELECT ALGORITHM")
    print("  " + "-" * 60)
    print("  1. BFS     — Closest songs first (safe picks)")
    print("  2. DFS     — Deep exploration (niche discoveries)")
    print("  3. A*      — Smartest match to your target vibe")
    print("  4. Compare — Run all 3 and compare results")
    print("  " + "-" * 60)
    while True:
        try:
            choice = int(input("\n  Enter algorithm number: "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("  Enter a number between 1 and 4.")
        except ValueError:
            print("  Please enter a valid number.")



#  STEP 5 — BFS
#  Explores level by level (closest songs first)


def bfs_recommend(G, seed_id, top_n=5):
    visited = set()
    queue   = deque()
    results = []

    queue.append((seed_id, 0))
    visited.add(seed_id)

    while queue and len(results) < top_n:
        current, hops = queue.popleft()
        neighbours = sorted(
            G[current].items(),
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        for neighbour, data in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, hops + 1))
                if neighbour != seed_id:
                    results.append({
                        "song_id" : neighbour,
                        "score"   : data["weight"],
                        "hops"    : hops + 1
                    })
                if len(results) >= top_n:
                    break
    return results



#  STEP 6 — DFS
#  Explores deep first ( niche unexpected songs)


def dfs_recommend(G, seed_id, top_n=5):
    visited = set()
    results = []

    def dfs(node, hops):
        if len(results) >= top_n:
            return
        visited.add(node)
        neighbours = sorted(
            G[node].items(),
            key=lambda x: x[1]["weight"]
        )
        for neighbour, data in neighbours:
            if neighbour not in visited:
                if neighbour != seed_id:
                    results.append({
                        "song_id" : neighbour,
                        "score"   : data["weight"],
                        "hops"    : hops + 1
                    })
                dfs(neighbour, hops + 1)
                if len(results) >= top_n:
                    return

    dfs(seed_id, 0)
    return results


#  STEP 7 — A*
#  Uses target vibe as heuristic ( smartest match)


def heuristic(G, node_id, target_vibe):
    node_vibes = get_song(G, node_id)["vibe"]
    return 0 if target_vibe in node_vibes else 5

def astar_recommend(G, seed_id, target_vibe, top_n=5):
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
            song      = get_song(G, current)
            why_parts = []
            if target_vibe in song["vibe"]:
                why_parts.append(f"{target_vibe} vibe")
            if song["bpm_range"]:
                why_parts.append(f"{song['bpm_range']} BPM")
            if song["city"]:
                why_parts.append(f"{song['city']} scene")
            results.append({
                "song_id" : current,
                "score"   : path_score,
                "hops"    : hops,
                "why"     : " + ".join(why_parts) if why_parts else "similar features"
            })

        for neighbour, data in G[current].items():
            if neighbour not in visited:
                edge_weight = data["weight"]
                h           = heuristic(G, neighbour, target_vibe)
                new_cost    = cost + (10 - edge_weight) + h
                heapq.heappush(heap, (new_cost, neighbour, edge_weight, hops + 1))

    return results



#  STEP 8 — PRINT RESULTS


def print_bfs(G, seed_id):
    print("\n  " + "=" * 62)
    print("  BFS — Closest songs first (safe picks)")
    print("  " + "=" * 62)
    for i, r in enumerate(bfs_recommend(G, seed_id), 1):
        print(f"  {i}. {song_label(G, r['song_id']):<45} "
              f"[score: {r['score']}]  {r['hops']} hop(s)")
    print()

def print_dfs(G, seed_id):
    print("\n  " + "=" * 62)
    print("  DFS — Deep exploration (niche discoveries)")
    print("  " + "=" * 62)
    for i, r in enumerate(dfs_recommend(G, seed_id), 1):
        print(f"  {i}. {song_label(G, r['song_id']):<45} "
              f"[score: {r['score']}]  {r['hops']} hop(s)")
    print()

def print_astar(G, seed_id, target_vibe):
    print("\n  " + "=" * 62)
    print(f"  A* — Smartest match to '{target_vibe}' vibe")
    print("  " + "=" * 62)
    for i, r in enumerate(astar_recommend(G, seed_id, target_vibe), 1):
        print(f"  {i}. {song_label(G, r['song_id']):<45} "
              f"[score: {r['score']}]  why: {r['why']}")
    print()

def print_comparison(G, seed_id, target_vibe):
    print("\n  " + "=" * 62)
    print("  RAPMAP — Algorithm Comparison")
    print("  " + "=" * 62)
    print(f"  Seed song   : {song_label(G, seed_id)}")
    print(f"  Target vibe : {target_vibe}")
    print_bfs(G, seed_id)
    print_dfs(G, seed_id)
    print_astar(G, seed_id, target_vibe)
    print("  " + "=" * 62)
    print("  SUMMARY")
    print("  " + "=" * 62)
    print("  BFS → Safe, expected picks close to your seed song")
    print("  DFS → Adventurous, niche, unexpected discoveries")
    print(f"  A*  → Smartest match targeting '{target_vibe}' vibe")
    print("  " + "=" * 62 + "\n")



#  STEP 9 — VISUALIZE GRAPH


def visualize_graph(G, df, seed_id):
    print("  Building graph visualization...")
    plt.figure(figsize=(18, 12))
    plt.title("RapMap — Desi Hip Hop Song Similarity Graph",
              fontsize=16, fontweight="bold", pad=20)

    pos = nx.spring_layout(G, seed=42, k=0.9)

    city_colors = {
        "Delhi"      : "#9b59b6",
        "Mumbai"     : "#e74c3c",
        "Pune"       : "#2ecc71",
        "Chandigarh" : "#f39c12",
        "Other"      : "#95a5a6"
    }

    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        city = df.loc[df["song_id"] == node, "city"].values[0]
        node_colors.append(city_colors.get(city, "#95a5a6"))
        node_sizes.append(900 if node == seed_id else 350)

    edge_weights = [G[u][v]["weight"] * 0.25 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, width=edge_weights,
                           alpha=0.12, edge_color="#555555")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_nodes(G, pos, nodelist=[seed_id],
                           node_color="none", edgecolors="#000000",
                           node_size=1000, linewidths=2.5)
    nx.draw_networkx_labels(
        G, pos,
        labels={n: df.loc[df["song_id"] == n, "title"].values[0]
                for n in G.nodes()},
        font_size=6.5, font_color="#1a1a1a"
    )

    legend_patches = [
        mpatches.Patch(color=c, label=city)
        for city, c in city_colors.items()
    ]
    legend_patches.append(
        mpatches.Patch(facecolor="white", edgecolor="black",
                       linewidth=1.5, label="★ Seed song")
    )
    plt.legend(handles=legend_patches, loc="upper left",
               title="City Scene", fontsize=9, title_fontsize=10)
    plt.axis("off")
    plt.tight_layout()

    output_path = r"D:\STUDY MATERIALS\VIT\Second sem\vityarthi project\rapmap_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Graph saved to: {output_path}")



#  STEP 10 — MOST CONNECTED SONGS

def print_top_connected(G, df, top_n=5):
    print(f"\n  Top {top_n} most connected songs:")
    connections = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    for song_id, degree in connections[:top_n]:
        title  = df.loc[df["song_id"] == song_id, "title"].values[0]
        artist = df.loc[df["song_id"] == song_id, "artist"].values[0]
        print(f"  {title} — {artist}  ({degree} connections)")


#  MAIN — Interactive loop


if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  RAPMAP — Desi Hip Hop Recommender")
    print("  Built for CSA2001 — Fundamentals in AI and ML")
    print("=" * 62 + "\n")

    df = load_data(CSV_FILE)
    G  = build_graph(df, threshold=5)
    print_top_connected(G, df)

    while True:
        print("\n" + "=" * 62)
        print("  MAIN MENU")
        print("=" * 62)
        print("  1. Get song recommendations")
        print("  2. Visualize the song graph")
        print("  3. Show most connected songs")
        print("  4. Exit")
        print("=" * 62)

        try:
            menu_choice = int(input("\n  Enter your choice: "))
        except ValueError:
            print("  Please enter a valid number.")
            continue

        if menu_choice == 1:
            seed_id     = pick_song(df)
            target_vibe = pick_vibe()
            algo_choice = pick_algorithm()

            if algo_choice == 1:
                print_bfs(G, seed_id)
            elif algo_choice == 2:
                print_dfs(G, seed_id)
            elif algo_choice == 3:
                print_astar(G, seed_id, target_vibe)
            elif algo_choice == 4:
                print_comparison(G, seed_id, target_vibe)

        elif menu_choice == 2:
            seed_id = pick_song(df)
            visualize_graph(G, df, seed_id)

        elif menu_choice == 3:
            print_top_connected(G, df)

        elif menu_choice == 4:
            print("\n  Thanks for using RapMap. Keep it real.\n")
            break

        else:
            print("  Please enter a number between 1 and 4.")
