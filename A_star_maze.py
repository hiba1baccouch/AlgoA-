import heapq
import matplotlib.pyplot as plt
import random
import sys

# Increase recursion depth for deep mazes
sys.setrecursionlimit(5000)

# ---------------------------
# MAZE GENERATION (Recursive Backtracker)
# ---------------------------
def generate_maze(rows, cols):
    # Grid: 1 = wall, 0 = path
    # Initialize with all walls
    maze = [[1 for _ in range(cols)] for _ in range(rows)]

    def get_neighbors(r, c):
        neighbors = []
        # Check 2 steps away (skipping walls)
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc, dr // 2, dc // 2))
        random.shuffle(neighbors)
        return neighbors

    def visit(r, c):
        maze[r][c] = 0 # Mark as path
        for nr, nc, dr, dc in get_neighbors(r, c):
            if maze[nr][nc] == 1: # If unvisited
                maze[r + dr][c + dc] = 0 # Knock down wall between
                visit(nr, nc)

    # Start from (0, 0)
    visit(0, 0)
    
    # Ensure start and goal are open (though recursive backtracker usually handles this if we start there)
    maze[0][0] = 0
    maze[rows-1][cols-1] = 0
    
    return maze

# ---------------------------
# A* ALGORITHM
# ---------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    steps = []

    while open_set:
        current_f, current = heapq.heappop(open_set)
        current_g = g_score[current]
        current_h = heuristic(current, goal)
        
        neighbors_details = []

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            steps.append({
                'node': current,
                'g': current_g,
                'h': current_h,
                'f': current_f,
                'neighbors': [],
                'status': 'Goal Reached!'
            })
            return list(reversed(path)), steps

        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tentative_g = g_score[current] + 1
                h_val = heuristic((nx, ny), goal)
                f_val = tentative_g + h_val
                
                neighbors_details.append(((nx, ny), tentative_g, h_val, f_val))

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    heapq.heappush(open_set, (f_val, (nx, ny)))
                    came_from[(nx, ny)] = current
        
        steps.append({
            'node': current,
            'g': current_g,
            'h': current_h,
            'f': current_f,
            'neighbors': neighbors_details,
            'status': 'Searching...'
        })

    return None, steps

# ---------------------------
# VISUALIZATION
# ---------------------------
def visualize_maze_search(grid, start, goal, path, steps):
    rows, cols = len(grid), len(grid[0])
    
    fig = plt.figure(figsize=(16, 8))
    ax_grid = plt.subplot2grid((1, 2), (0, 0))
    ax_text = plt.subplot2grid((1, 2), (0, 1))
    
    # Plot Maze
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [-0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax_grid.imshow(grid, cmap=cmap, norm=norm)
    
    # Start/Goal
    ax_grid.plot(start[1], start[0], 'go', markersize=8, label='Start')
    ax_grid.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')
    
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title("Labyrinth A* Search")

    # Dynamic elements
    visited_plot, = ax_grid.plot([], [], 'bo', markersize=4, alpha=0.6, label='Visited')
    path_plot, = ax_grid.plot([], [], 'y-', linewidth=2, label='Path')
    current_node_plot, = ax_grid.plot([], [], 'cyan', marker='s', markersize=6, linestyle='None', label='Current')

    ax_text.axis('off')
    text_display = ax_text.text(0, 1, "", va='top', fontsize=10, family='monospace')

    state = {'step_idx': -1}

    def update_display(idx):
        if idx < 0: return

        if idx < len(steps):
            step = steps[idx]
            node = step['node']
            
            # Visited nodes
            current_visited_x = [steps[i]['node'][1] for i in range(idx + 1)]
            current_visited_y = [steps[i]['node'][0] for i in range(idx + 1)]
            visited_plot.set_data(current_visited_x, current_visited_y)
            
            # Highlight current node
            current_node_plot.set_data([node[1]], [node[0]])
            
            # Text Info
            info = f"Step {idx+1}: {step['status']}\n"
            info += f"--------------------------------\n"
            info += f"Current Node: {node}\n"
            info += f"  G (Cost): {step['g']}\n"
            info += f"  H (Dist): {step['h']}\n"
            info += f"  F (Total): {step['f']}\n\n"
            
            if step['neighbors']:
                info += "Neighbors Checked:\n"
                for n_node, ng, nh, nf in step['neighbors']:
                    info += f"  {n_node}: G={ng}, H={nh}, F={nf}\n"
            else:
                info += "No valid neighbors.\n"
            
            text_display.set_text(info)
            ax_grid.set_title(f"Visiting: {node}")
            
        else:
            # Path Phase
            if path:
                path_idx = idx - len(steps)
                if path_idx < len(path):
                    current_path_x = [path[i][1] for i in range(path_idx + 1)]
                    current_path_y = [path[i][0] for i in range(path_idx + 1)]
                    path_plot.set_data(current_path_x, current_path_y)
                    
                    ax_grid.set_title("Tracing Final Path")
                    text_display.set_text("Search Complete.\n\nTracing back path from Goal to Start.\n\nThe yellow line represents the\noptimal path found.")

    def on_key(event):
        if event.key in [' ', 'right']:
            state['step_idx'] += 1
            max_steps = len(steps) + (len(path) if path else 0)
            if state['step_idx'] >= max_steps:
                state['step_idx'] = max_steps - 1
            update_display(state['step_idx'])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    text_display.set_text("LABYRINTH MODE\n\nPress SPACE or RIGHT ARROW\nto advance.\n\nDetailed values shown here.")
    ax_grid.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

# ---------------------------
# MAIN
# ---------------------------
# Maze dimensions (must be odd for recursive backtracker to look nice with walls)
ROWS, COLS = 21, 21 
maze = generate_maze(ROWS, COLS)

start = (0, 0)
goal = (ROWS-1, COLS-1)

# Ensure start/goal are accessible (0)
maze[start[0]][start[1]] = 0
maze[goal[0]][goal[1]] = 0

print("Solving maze...")
path, steps = astar(maze, start, goal)

if path:
    print(f"Path found! Length: {len(path)}")
    visualize_maze_search(maze, start, goal, path, steps)
else:
    print("No path found!")
