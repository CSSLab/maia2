from .utils import *
from .main import *

def preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict):
        
    if fen.split(' ')[1] == 'w':
        board = chess.Board(fen)
    elif fen.split(' ')[1] == 'b':
        board = chess.Board(fen).mirror()
    else:
        raise ValueError(f"Invalid fen: {fen}")
        
    board_input = board_to_tensor(board)
    
    elo_self = map_to_category(elo_self, elo_dict)
    elo_oppo = map_to_category(elo_oppo, elo_dict)
    
    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor([all_moves_dict[move.uci()] for move in board.legal_moves])
    legal_moves[legal_moves_idx] = 1
    
    return board_input, elo_self, elo_oppo, legal_moves


class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, all_moves_dict, elo_dict):
        
        self.all_moves_dict = all_moves_dict
        self.data = data.values.tolist()
        self.elo_dict = elo_dict
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        fen, _, elo_self, elo_oppo = self.data[idx]

        board_input, elo_self, elo_oppo, legal_moves = preprocessing(fen, elo_self, elo_oppo, self.elo_dict, self.all_moves_dict)
        
        return fen, board_input, elo_self, elo_oppo, legal_moves

def get_preds(model, dataloader, all_moves_dict_reversed):
    
    move_probs = []
    win_probs = []
    
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        
        for fens, boards, elos_self, elos_oppo, legal_moves in dataloader:
            
            boards = boards.to(device)
            elos_self = elos_self.to(device)
            elos_oppo = elos_oppo.to(device)
            legal_moves = legal_moves.to(device)

            logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia + legal_moves.log()
            probs = logits_maia_legal.softmax(dim=-1).cpu().tolist()
            
            logits_value = (logits_value / 2 + 0.5).clamp(0, 1).cpu().tolist()
        
            for i in range(len(fens)):
                
                fen = fens[i]
                black_flag = False
                
                # calculate win probability
                logit_value = logits_value[i]
                if fen.split(" ")[1] == "b":
                    logit_value = 1 - logit_value
                    black_flag = True
                win_probs.append(round(logit_value, 4))
                
                # calculate move probabilities
                move_probs_each = {}
                legal_move_indices = legal_moves[i].nonzero().flatten().cpu().numpy().tolist()
                legal_moves_mirrored = []
                for move_idx in legal_move_indices:
                    move = all_moves_dict_reversed[move_idx]
                    if black_flag:
                        move = mirror_move(move)
                    legal_moves_mirrored.append(move)
                
                for j in range(len(legal_move_indices)):
                    move_probs_each[legal_moves_mirrored[j]] = round(probs[i][legal_move_indices[j]], 4)
                
                move_probs_each = dict(sorted(move_probs_each.items(), key=lambda item: item[1], reverse=True))
                move_probs.append(move_probs_each)
    
    return move_probs, win_probs


def inference_batch(data, model, verbose, batch_size, num_workers):

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    dataset = TestDataset(data, all_moves_dict, elo_dict)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, 
                                            drop_last=False,
                                            num_workers=num_workers)
    if verbose:
        dataloader = tqdm.tqdm(dataloader)
        
    move_probs, win_probs = get_preds(model, dataloader, all_moves_dict_reversed)
    
    data["win_probs"] = win_probs
    data["move_probs"] = move_probs
    
    acc = 0
    for i in range(len(data)):
        highest_prob_move = max(move_probs[i], key=move_probs[i].get)
        if highest_prob_move == data.iloc[i]["move"]:
            acc += 1
    acc = round(acc / len(data), 4)
    
    return data, acc


def prepare():

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    
    return [all_moves_dict, elo_dict, all_moves_dict_reversed]


def inference_each(model, prepared, fen, elo_self, elo_oppo):
    
    all_moves_dict, elo_dict, all_moves_dict_reversed = prepared
    
    board_input, elo_self, elo_oppo, legal_moves = preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict)
    
    device = next(model.parameters()).device
    
    model.eval()
    
    board_input = board_input.unsqueeze(dim=0).to(device)
    elo_self = torch.tensor([elo_self]).to(device)
    elo_oppo = torch.tensor([elo_oppo]).to(device)
    legal_moves = legal_moves.unsqueeze(dim=0).to(device)
    
    logits_maia, _, logits_value = model(board_input, elo_self, elo_oppo)
    logits_maia_legal = logits_maia + legal_moves.log()
    probs = logits_maia_legal.softmax(dim=-1).cpu().tolist()
    
    logits_value = (logits_value / 2 + 0.5).clamp(0, 1).item()
    
    black_flag = False
    if fen.split(" ")[1] == "b":
        logits_value = 1 - logits_value
        black_flag = True
    win_prob = round(logits_value, 4)
    
    move_probs = {}
    legal_move_indices = legal_moves.nonzero().flatten().cpu().numpy().tolist()
    legal_moves_mirrored = []
    for move_idx in legal_move_indices:
        move = all_moves_dict_reversed[move_idx]
        if black_flag:
            move = mirror_move(move)
        legal_moves_mirrored.append(move)
    
    for j in range(len(legal_move_indices)):
        move_probs[legal_moves_mirrored[j]] = round(probs[0][legal_move_indices[j]], 4)
    
    move_probs = dict(sorted(move_probs.items(), key=lambda item: item[1], reverse=True))
    
    return move_probs, win_prob

