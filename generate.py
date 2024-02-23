from engine import best_move, generate_all_legal_fens
from data import save_to_json, load_from_json, add_start_pos_suffix

fens = generate_all_legal_fens(5)

ground_truth_size_5 = {}

print(f"Processing {len(fens)} fens!")
for i, fen in enumerate(fens):
    res = best_move(add_start_pos_suffix(fen))
    ground_truth_size_5[fen] = res

    if i % 350 == 0:
        print(f"Processed {i} fens")
        save_to_json(ground_truth_size_5, "__mappings_size_5.json")

save_to_json(ground_truth_size_5, "__mappings_size_5.json")
print("Done!")
