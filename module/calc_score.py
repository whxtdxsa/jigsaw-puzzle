import numpy as np
from tqdm import tqdm
import torch

def calc_puzzle(answer_df, submission_df):
    # Check for missing values in submission_df
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")

    # Public or Private answer Sample and Sorting by 'ID'
    submission_df = submission_df[submission_df.iloc[:, 0].isin(answer_df.iloc[:, 0])]
    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)

    # Check for length in submission_df
    if len(submission_df) != len(answer_df):
        raise ValueError("The submission dataframe wrong length.")

    # Convert position data to numpy arrays for efficient computation
    answer_positions = answer_df.iloc[:, 2:].to_numpy()  # Excluding ID, img_path, and type columns
    submission_positions = submission_df.iloc[:, 1:].to_numpy()  # Excluding ID column

    # Initialize the dictionary to hold accuracies
    accuracies = {}

    # Define combinations for 2x2 and 3x3 puzzles
    combinations_2x2 = [(i, j) for i in range(3) for j in range(3)]
    combinations_3x3 = [(i, j) for i in range(2) for j in range(2)]

    # 1x1 Puzzle Accuracy
    accuracies['1x1'] = np.mean(answer_positions == submission_positions)

    # Calculate accuracies for 2x2, 3x3, and 4x4 puzzles
    for size in range(2, 5):  # Loop through sizes 2, 3, 4
        correct_count = 0  # Initialize counter for correct full sub-puzzles
        total_subpuzzles = 0

        # Iterate through each sample's puzzle
        for i in range(len(answer_df)):
            puzzle_a = answer_positions[i].reshape(4, 4)
            puzzle_s = submission_positions[i].reshape(4, 4)
            combinations = combinations_2x2 if size == 2 else combinations_3x3 if size == 3 else [(0, 0)]

            # Calculate the number of correct sub-puzzles for this size within a 4x4
            for start_row, start_col in combinations:
                rows = slice(start_row, start_row + size)
                cols = slice(start_col, start_col + size)
                if np.array_equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
                    correct_count += 1
                total_subpuzzles += 1

        accuracies[f'{size}x{size}'] = correct_count / total_subpuzzles

    score = (accuracies['1x1'] + accuracies['2x2'] + accuracies['3x3'] + accuracies['4x4']) / 4.
    return score

def eval_model(model, dataloader, df):
    outs = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader):
            x = x.to('cuda')
            out = model.forward_test(x)
            out = out.argmax(dim=2).cpu().numpy()
            outs.append(out)
    outs = np.vstack(outs)

    pred_df = df.copy().drop(columns=['img_path'])
    
    for I, (idx, row) in enumerate(tqdm(pred_df.iterrows(), total=len(df))):
        w = outs[I].reshape(24,24)
        CNT_ROW = np.zeros((4,4,4), dtype=np.int32)
        CNT_COL = np.zeros((4,4,4), dtype=np.int32)
        for i in range(24):
            for j in range(24):
                ROW = i // 6
                COL = j // 6
                v = w[i][j]
                CNT_ROW[ROW][COL][v // 24 // 6] += 1
                CNT_COL[ROW][COL][v % 24 // 6] += 1
        ans = CNT_ROW.argmax(2) * 4 + CNT_COL.argmax(2) + 1
        ans = ans.reshape(16)
        ans = list(map(str, ans))
        pred_df.loc[idx, '1':'16'] = ans

    return pred_df
