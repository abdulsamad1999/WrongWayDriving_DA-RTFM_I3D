import numpy as np

def generate_gt_from_list(list_file_path, output_npy_path):
    labels = []
    with open(list_file_path, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            labels.append(int(label))

    labels = np.array(labels, dtype=np.int32)
    np.save(output_npy_path, labels)
    print(f"✅ Ground truth saved at: {output_npy_path} | Total labels: {len(labels)}")

# Usage
generate_gt_from_list('list/mytest.list', 'list/gt-wrongway.npy')
