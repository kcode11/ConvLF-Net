import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

def eval():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'saved_model.pkl'
    # KIR Dataset
    model = ConvLF_Net(channel_in=13, num_classes=6).to(device)
    # CRWU dataset
    # model = ConvLF_Net(channel_in = 1,num_classes = 4).to(device)
    model.load_state_dict(torch.load(model_path))
    print("[Info]: Finish creating model!", flush=True)

    label_total = []
    preds = []
    test_acc = 0.0
    final_outputs = []

    model.eval()
    test_pbar = tqdm(test_loader, position=0, leave=True)
    with torch.no_grad():
        for data, labels in test_pbar:
            data = data.float().to(device)
            outputs = model(data)
            _, test_pred = torch.max(outputs, 1)
            preds.append(test_pred.cpu().detach())

            labels = labels.to(device)
            label_total.append(labels.cpu().detach())

            final_outputs.append(outputs.cpu().detach())

            test_acc += (test_pred == labels).float().mean().item()

        print("test_acc:", test_acc / len(test_loader))

    return label_total, preds, final_outputs

labels, preds, final_outputs = eval()

# calculate conf_matrix
conf_matrix = confusion_matrix(torch.cat(labels).numpy(), torch.cat(preds).numpy())

# TP、FN、FP、TN
num_classes = conf_matrix.shape[0]
tp = np.diag(conf_matrix)  # True Positives for each class
fn = np.sum(conf_matrix, axis=1) - tp  # False Negatives for each class
fp = np.sum(conf_matrix, axis=0) - tp  # False Positives for each class
tn = np.sum(conf_matrix) - (fp + fn + tp)  # True Negatives for each class

# metric
accuracy = np.sum(tp) / np.sum(conf_matrix)
precision = np.mean(tp / (tp + fp))  # Precision
recall = np.mean(tp / (tp + fn))  # Recall
f1 = 2 * (precision * recall) / (precision + recall)  # F1 Score
specificity = np.mean(tn / (tn + fp))  # Specificity

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

final_outputs = torch.cat(final_outputs)

# t-SNE 
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(final_outputs.numpy())
unique_labels = np.unique(torch.cat(labels).numpy())
palette = sns.color_palette("Set1", n_colors=len(unique_labels))

plt.figure(figsize=(15, 6))
# visualize TSNE
plt.subplot(1, 2, 1)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=torch.cat(labels).numpy(), palette=palette, legend='full')
plt.title("t-SNE Visualization")
plt.savefig("t-SNE_Visualization.png")

# visualize Confusion_Matrix
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig("Confusion_Matrix.png")

plt.show()
