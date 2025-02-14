import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from properscoring import crps_ensemble
import matplotlib.pyplot as plt

# Helper function to compute Winkler Score
def winkler_score(forecast, lower, upper, observation, alpha=0.1):
    delta = upper - lower
    if observation < lower:
        return delta + 2 * (lower - observation) / alpha
    elif observation > upper:
        return delta + 2 * (observation - upper) / alpha
    else:
        return delta

def parse_log_file(file_path):
    outputs = []
    groundtruths = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    regex_pattern = r"[-+]?\d*\.\d+(?:e[-+]?\d+)?|\d+"
    
    for i, line in enumerate(lines):
        if 'Output (sample):' in line:
            current_output = []
            for b in range(4):  # 4 batches
                for v in range(17):  # 16 values per batch
                    if v != 16:
                        current_output.extend(re.findall(regex_pattern, lines[i+v+b*17]))
            outputs.append([float(num) for num in current_output])
        elif 'Groundtruth (sample):' in line:
            current_gt = []
            for b in range(4):  # 4 batches
                for v in range(17):  # 16 values per batch
                    if v != 16:
                        current_gt.extend(re.findall(regex_pattern, lines[i+v+b*17]))
            groundtruths.append([float(num) for num in current_gt])

    return outputs, groundtruths

def calculate_spm_rmse(all_outputs, all_groundtruths):
    # 假设实际的 P(t) 是 all_groundtruths
    # 计算 Pclr(t) (理想情况下的晴天输出)，可以根据一些近似方法来获取
    
    # 简单的近似假设 Pclr(t) 作为一个常数或者随时间变化的某种函数
    Pclr = np.full_like(all_groundtruths, np.mean(all_groundtruths))  # 这是一个简单的假设，实际中应基于太阳角度和天气数据计算
    
    # 计算每个时间点的相对输出 kclr
    kclr_actual = all_groundtruths / Pclr
    
    # 假设 kclr 在 t 到 t+T 保持恒定，P(t+T) = kclr * Pclr(t+T)
    predicted_spm = kclr_actual[:-1] * Pclr[1:]
    
    # 计算 SPM 的 RMSE
    spm_rmse = np.sqrt(mean_squared_error(all_groundtruths[1:], predicted_spm))
    return spm_rmse


def calculate_metrics(outputs, groundtruths):
    rmses = []
    maes = []
    fss = []
    crpss = []
    wss = []

    all_outputs = np.concatenate(outputs)
    all_groundtruths = np.concatenate(groundtruths)

    all_outputs = all_outputs[-5000:]
    all_groundtruths = all_groundtruths[-5000:]

    print(1)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(all_groundtruths, all_outputs))
    mae = mean_absolute_error(all_groundtruths, all_outputs)
    
    # Calculate Forecast Skill (FS)
    # spm_rmse = calculate_spm_rmse(all_outputs, all_groundtruths)
    # fs = 1 - (rmse**2 / spm_rmse**2) if spm_rmse != 0 else np.nan
    spm_rmse = np.sqrt(mean_squared_error(all_groundtruths[1500:], all_groundtruths[:-1500]))  # Simple persistence model
    fs = 1 - (rmse**2 / spm_rmse**2) if spm_rmse != 0 else np.nan

    # Calculate CRPS
    ensemble_forecast = np.tile(all_outputs[:30], (len(all_groundtruths[:30]), 1))
    crps_value = crps_ensemble(all_groundtruths[:30], ensemble_forecast).mean()

    alpha = 0.05
    std_dev = np.std(all_outputs)
    lower = all_outputs - 1.96 * std_dev  # Adjusted for 95% confidence interval
    upper = all_outputs + 1.96 * std_dev  # Adjusted for 95% confidence interval
    ws_value = np.mean([winkler_score(o, l, u, g, alpha) for o, l, u, g in zip(all_outputs, lower, upper, all_groundtruths)])
    # Append metrics to lists for averaging
    rmses.append(rmse)
    maes.append(mae)
    fss.append(fs)
    crpss.append(crps_value)
    wss.append(ws_value)

    avg_rmse = np.mean(rmses) if rmses else np.nan
    avg_mae = np.mean(maes) if maes else np.nan
    avg_fs = np.mean(fss) if fss else np.nan
    avg_crps = np.mean(crpss) if crpss else np.nan
    avg_ws = np.mean(wss) if wss else np.nan

    return avg_rmse, avg_mae, avg_fs, avg_crps, avg_ws, all_outputs, all_groundtruths

def visualize_predictions(predictions, groundtruths, filename='pred_vs_gt.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions[-50000:], label='Predicted')
    plt.plot(groundtruths[-50000:], label='Ground Truth')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predicted vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    log_filename = 'train_result.log'
    outputs, groundtruths = parse_log_file(log_filename)
    
    if not outputs or not groundtruths:
        print("No valid output or groundtruth data found in the log file.")
    else:
        avg_rmse, avg_mae, avg_fs, avg_crps, avg_ws, all_outputs, all_groundtruths = calculate_metrics(outputs, groundtruths)
        
        print(f"Average RMSE: {avg_rmse}")
        print(f"Average MAE: {avg_mae}")
        print(f"Average FS: {avg_fs}")
        print(f"Average CRPS: {avg_crps}")
        print(f"Average Winkler Score: {avg_ws}")
        
        visualize_predictions(all_outputs, all_groundtruths)
