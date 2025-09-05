
import random

def manhattan(a,b): 
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def choose_free_cell(grid, rnd):
    h=len(grid); w=len(grid[0])
    for _ in range(10000):
        y=rnd.randrange(1,h-1); x=rnd.randrange(1,w-1)
        if grid[y][x]==0:
            return (y,x)
    return (1,1)

def simulate_episode(grid, agent_count, density, comm, disturb, heuristic, seed):
    rnd = random.Random(seed)
    starts=[]; goals=[]
    for i in range(agent_count):
        s=choose_free_cell(grid, rnd)
        g=choose_free_cell(grid, rnd)
        starts.append(s); goals.append(g)
    avg_pair = sum(manhattan(s,g) for s,g in zip(starts,goals))/max(1,agent_count)

    base = 1.0 + 4.0*density + 0.01*agent_count
    comm_factor = {"none":1.1,"limited":1.05,"full":1.0}.get(comm,1.05)
    dist_factor = {"none":1.0,"moderate":1.15,"high":1.35}.get(disturb,1.0)
    heur_factor = {"admissible":1.0,"inadmissible":0.95,"learned":0.9,"hybrid":0.88}.get(heuristic,0.95)

    expansions = int((base*dist_factor*comm_factor)*1000*(1.0/heur_factor)*(0.9+0.2*rnd.random()))
    makespan = int(avg_pair*dist_factor*(0.9+0.3*rnd.random())*(1.05 if density>0.3 else 1.0))
    sum_costs = int(makespan*agent_count*(0.9+0.2*rnd.random()))
    runtime_ms = int(expansions*(0.5+0.5*rnd.random()))
    replans = int((dist_factor-1.0)*10*(0.6+rnd.random()))
    collisions = int(max(0,(density*agent_count*0.02-0.2)*(1.0/heur_factor)+rnd.random()-0.5))
    memory_mb = round(50 + 0.01*expansions + 0.2*agent_count, 2)

    success_score = 0.75*(1.0/comm_factor)*(1.0/(dist_factor))*(heur_factor/1.0)*(1.1-0.5*density)
    success = 1 if rnd.random() < max(0.05, min(0.98, success_score)) else 0

    features = {
        "start_goal_dispersion": avg_pair,
        "open_list_pressure_est": int(expansions/max(1,agent_count))
    }
    metrics = {
        "success": success,
        "makespan": makespan,
        "sum_of_costs": sum_costs,
        "runtime_ms": runtime_ms,
        "node_expansions": expansions,
        "replans": replans,
        "collisions": collisions,
        "memory_mb": memory_mb
    }
    return metrics, features
