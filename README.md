# RapMap — Desi Hip Hop Song Recommender

RapMap is a graph-based song recommendation system built specifically for Desi Hip Hop. It models songs as nodes in a graph, connects them based on similarity, and uses three search algorithms — BFS, DFS, and A* — to generate recommendations.

Built as part of CSA2001 — Fundamentals in AI and ML at VIT Bhopal.

---

## Why RapMap?

Mainstream platforms like Spotify and Apple Music fail at recommending Desi Hip Hop because:
- Their models are trained on Western music data
- Indian artists get mislabeled under Bollywood or World Music
- They do not understand cultural context — city scenes, collab networks, or language nuance in Hindi, Punjabi, Haryanvi, and Urdu rap

RapMap solves this by building a hand-curated dataset of Desi Hip Hop songs and using graph search algorithms to find meaningful recommendations.

---

## Project Structure

```
rapmap/
├── rapmap.py           # Main program — all algorithms and menu
├── DHH Database.csv    # Hand-curated dataset of 60 Desi Hip Hop songs
└── README.md           # This file
```

---

## Dataset

The dataset (`DHH Database.csv`) contains 60 Desi Hip Hop songs with the following fields:

| Field | Description | Example |
|---|---|---|
| song_id | Unique identifier | 1 |
| title | Song title | Maina |
| artist | Main artist | Seedhe Maut |
| language | Language(s) of the song | Hindi, Punjabi |
| vibe | Mood/feel of the song | Aggressive, Conscious |
| bpm_range | Tempo category | slow / mid / fast |
| city | City the artist is from | Delhi |
| featured_artists | Featured artists if any | Krsna |

Artists covered include: Seedhe Maut, Krsna, Divine, Raftaar, MC Stan, Emiway, Prabh Deep, Yashraj, Naam Sujal, Ikka, and more.

---

## How It Works

### 1. Graph Construction
Each song is a node. Two songs are connected by an edge if their similarity score is 5 or above. The similarity score is calculated as:

```
Same language     → +3 points
Same vibe         → +3 points
Same BPM range    → +2 points
Same city         → +2 points
Same artist       → +4 points
Collab together   → +4 points
```

### 2. Search Algorithms

| Algorithm | Strategy | Result type |
|---|---|---|
| BFS | Explores level by level from seed song | Safe, expected, close picks |
| DFS | Dives deep along one path first | Niche, unexpected, adventurous |
| A* | Uses target vibe as heuristic | Smartest match to your mood |

---

## Requirements

- Python 3.8 or above
- pip (comes with Python)

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| pandas | latest | Reading and cleaning the CSV dataset |
| networkx | latest | Building and traversing the song graph |
| matplotlib | latest | Visualizing the graph |

---

## Setup and Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rapmap.git
cd rapmap
```

### Step 2 — Install dependencies

```bash
pip install pandas networkx matplotlib
```

No virtual environment is required. All three libraries are standard data science packages.

### Step 3 — Verify your files

Make sure both files are in the same folder:
```
rapmap/
├── rapmap.py
└── DHH Database.csv
```

### Step 4 — Open rapmap.py and update the CSV path

Find line 14 in `rapmap.py`:
```python
CSV_FILE = r"D:\STUDY MATERIALS\VIT\Second sem\vityarthi project\DHH Database.csv"
```

Change it to the path where you have cloned the project. For example:

**Windows:**
```python
CSV_FILE = r"C:\Users\YourName\rapmap\DHH Database.csv"
```

**Mac / Linux:**
```python
CSV_FILE = "DHH Database.csv"
```

If your terminal is already inside the `rapmap` folder, you can simply use:
```python
CSV_FILE = "DHH Database.csv"
```

---

## Running the Project

```bash
python rapmap.py
```

The program runs entirely in the terminal. No GUI setup is needed.

---

## Using the Menu

When you run the program, you will see:

```
================================================================
  RAPMAP — Desi Hip Hop Recommender
================================================================

  MAIN MENU
  1. Get song recommendations
  2. Visualize the song graph
  3. Show most connected songs
  4. Exit
```

### Option 1 — Get song recommendations

- A numbered list of all 60 songs is displayed
- Enter the number of the song you like
- Select a target vibe (Aggressive / Chill / Conscious / Motivational / Sad)
- Select an algorithm (BFS / DFS / A* / Compare all)
- Top 5 recommendations are displayed with scores and explanations

### Option 2 — Visualize the song graph

- Select a seed song
- A graph visualization opens showing all songs as nodes
- Nodes are colored by city scene
- The seed song is highlighted with a black outline
- The graph is saved as `rapmap_graph.png` in your project folder

### Option 3 — Show most connected songs

- Displays the top 5 songs with the most connections in the graph
- Useful for understanding which songs sit at the center of the Desi Hip Hop ecosystem

### Option 4 — Exit

Exits the program.

---

## Example Output

```
================================================================
  RAPMAP — Algorithm Comparison
================================================================
  Seed song   : Khatta flow — Seedhe Maut
  Target vibe : Aggressive

  BFS — Closest songs first (safe picks)
  1. Maina — Seedhe Maut          [score: 9]  1 hop(s)
  2. Do guna — Seedhe Maut        [score: 8]  1 hop(s)
  3. Woh raat — Raftaar           [score: 7]  2 hop(s)

  DFS — Deep exploration (niche discoveries)
  1. Blackball — SOS              [score: 5]  6 hop(s)
  2. New riot — Rebel             [score: 5]  8 hop(s)

  A* — Smartest match to 'Aggressive' vibe
  1. Woh raat — Raftaar           [score: 7]  why: Aggressive vibe + fast BPM + Delhi scene
  2. Akatsuki — Seedhe Maut       [score: 8]  why: Aggressive vibe + fast BPM + Delhi scene

  SUMMARY
  BFS → Safe, expected picks close to your seed song
  DFS → Adventurous, niche, unexpected discoveries
  A*  → Smartest match targeting 'Aggressive' vibe
```

---

## Updating the Dataset

The system is designed to scale. To add new songs:

1. Open `DHH Database.csv`
2. Add new rows following the existing format
3. Save the file
4. Run `rapmap.py` again — the graph rebuilds automatically

No code changes are needed when adding songs.

---

## Limitations

- Dataset is manually curated — 60 songs is enough to demonstrate the concept but a production system would need thousands
- Song features are manually tagged — a production system would use audio analysis APIs like Spotify's audio features endpoint
- No user feedback loop — the system does not learn from what you skip or replay
- City data is simplified — artists are assigned one primary city

---

## Future Scope

- Integrate Spotify API to pull real BPM, energy and danceability data automatically
- Add a user feedback loop where skipping a song reduces that edge weight over time
- Expand to other regional genres — Bengali hip hop, Tamil street rap, Bhojpuri rap
- Build a web interface using Flask

---

## Built With

- Python 3
- NetworkX
- Pandas
- Matplotlib

---

If you have questions or suggestions, feel free to reach out:
- GitHub: https://github.com/enchanter-coder
- Email: ishanshrivas0966@gmail.com(primary)
- Email: ishan.25bai10966@vitbhopal.ac.in(College mail)
  
