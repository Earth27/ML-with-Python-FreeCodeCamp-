import random

def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)
    
    if not opponent_history:
        return random.choice(['R', 'P', 'S'])
    
    counters = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    move_counts = {'R': 0, 'P': 0, 'S': 0}
    for move in opponent_history:
        move_counts[move] += 1
    
    most_common = max(move_counts, key=move_counts.get)
    
    if random.random() < 0.1:
        return random.choice(['R', 'P', 'S'])
    
    return counters[most_common]
