#! ./venv/bin/python
import numpy as np
import matplotlib.pyplot as plt

def expected_calibration_error(binsMap, countMap, confMap):
    """
    param: binsMap: a map containing the
    key dict pair of interval (eg. 0.1)
    with the total number of correct classifications
    in that bin

    param: countMap: a map containing the key dict
    pair of interval with total number of samples
    in that bin

    param: confMap: a map containing the key dict
    pair of interval with the average confidence
    score in that bin

    return : gapMap (haha): the key dict pair of
    interval with the confidence-accuracy gap
    """
    total_samples = 0
    ece = 0
    for k in list(countMap.keys()):
        total_samples += countMap[k]
    for k in list(binsMap.keys()):
        try:
            acc = binsMap[k]/countMap[k]
        except: 
            acc = 0
        conf = confMap[k]
        weight = countMap[k]/total_samples
        ece += weight * abs(acc - conf)
    return ece
    

def find_closest_bin(bins, val):
    """
    param: bins:

    param: val: 
    """
    binList = sorted(list(bins))
    diff = [abs(x - val) for x in binList]
    return binList[np.argmin(diff)]

def confidence_bins():
    """
    """
    bins = sorted(set(np.arange(0.0, 1.0, 0.1).round(1)))
    binsMap = dict.fromkeys(bins, 0)
    countMap = dict.fromkeys(bins, 0)
    confMap = dict.fromkeys(bins, 0)
    data = np.load('dump/calibrated_confidence_predictions.npy', allow_pickle=True).item()
    for k in data:
        closest = find_closest_bin(bins, data[k][0])
        countMap[closest] += 1
        confMap[closest] += data[k][0]
        if data[k][1] == True:
            binsMap[closest] += 1
    for k in list(confMap.keys()):
        try:
            confMap[k] /= countMap[k]
        except:
            confMap[k] = 0
    acc_y = []
    conf_x = []
    for k, v in binsMap.items():
        conf_x.append(round(k, 2))
        try:
            acc = binsMap[k]/countMap[k]
            acc_y.append(acc)
        except:
            acc_y.append(0)
    ece = expected_calibration_error(binsMap, countMap, confMap)
    fig, ax = plt.subplots()
    ax.text(0.3, 0.8, f'Expected Calibration Error: {round(ece, 2)}', fontsize = 10, bbox = dict(facecolor = 'red', alpha = 0.5), zorder=3)
    ax.bar(conf_x, acc_y, width=0.1, edgecolor='black', zorder=2, label='Outputs')
    ax.bar(conf_x, conf_x, width=0.1, alpha=0.4, zorder=1, color='red', edgecolor='red', label='Gap')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize='12')
    ax.grid(True)
    ax.set_title('Reliability Diagram for ResNet50 - Calibrated')
    ax.plot([0, 1], [0, 1], '--',transform=ax.transAxes)
    fig.tight_layout()
    plt.savefig('rd_calibrated.png') 


if __name__ == "__main__":
    confidence_bins()
