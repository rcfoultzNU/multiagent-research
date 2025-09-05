
import random

def generate_map(width: int, height: int, density: float, seed: int):
    rnd = random.Random(seed)
    grid = [[1 if rnd.random() < density else 0 for _ in range(width)] for _ in range(height)]
    def nsum(y, x):
        s=0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: 
                    continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < height and 0 <= nx < width:
                    s += grid[ny][nx]
        return s
    new = [[0]*width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            s = nsum(y,x)
            new[y][x] = 1 if (grid[y][x] and s>=4) or ((not grid[y][x]) and s>=6) else 0
    for x in range(width):
        new[0][x]=0; new[height-1][x]=0
    for y in range(height):
        new[y][0]=0; new[y][width-1]=0
    return new

def free_ratio(grid):
    h=len(grid); w=len(grid[0])
    return sum(1 for r in grid for v in r if v==0)/(w*h)

def clustering_index(grid):
    h=len(grid); w=len(grid[0])
    total=count=0
    for y in range(h):
        for x in range(w):
            if grid[y][x]==1:
                c=0
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dy==0 and dx==0: 
                            continue
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            c += 1 if grid[ny][nx]==1 else 0
                total+=c; count+=1
    return (total/count) if count else 0.0
