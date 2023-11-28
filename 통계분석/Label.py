import os
from collections import defaultdict
import matplotlib.pyplot as plt

# yolo annoation label 정보 확인
def parse_yolo_annotation(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        label_info = line.strip().split(' ')
        label = int(label_info[0])
        labels.append(label)

    return labels

# label 분포 정보 저장
def analyze_label_distribution(data_folder):
    label_counts = defaultdict(int)

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            annotation_path = os.path.join(data_folder, filename)
            labels = parse_yolo_annotation(annotation_path)

            for label in labels:
                label_counts[label] += 1

    return label_counts

# histogram으로 시각화
def plot_label_distribution(label_counts):
    labels, counts = zip(*label_counts.items())

    plt.bar(labels, counts, align='center')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.show()

# train
data_folder = os.path.join(output_folder, 'train')
label_counts = analyze_label_distribution(data_folder)
print("Train Label Distribution:", label_counts)
plot_label_distribution(label_counts)

# test
data_folder = os.path.join(output_folder, 'test')
label_counts = analyze_label_distribution(data_folder)
print("Test Label Distribution:", label_counts)
plot_label_distribution(label_counts)

# valid
data_folder = os.path.join(output_folder, 'valid')
label_counts = analyze_label_distribution(data_folder)
print("Valid Label Distribution:", label_counts)
plot_label_distribution(label_counts)
