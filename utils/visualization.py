from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_confusion_matrix_with_details(y_true, y_pred, class_names):
    """
    Plots a confusion matrix with numbers, percentages, and details of correct and incorrect predictions.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Prediction details
    correct_normals = cm[0, 0]
    incorrect_normals = cm[0, 1]
    correct_pneumonia = cm[1, 1]
    incorrect_pneumonia = cm[1, 0]

    # Print details
    print(f"Correct Normals: {correct_normals}")
    print(f"Incorrect Normals: {incorrect_normals}")
    print(f"Correct Pneumonia: {correct_pneumonia}")
    print(f"Incorrect Pneumonia: {incorrect_pneumonia}")

    # Confusion matrix with percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    # Plot the confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix with Details")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def visualize_class_distribution(train_df, val_df, test_df):
    """
    Displays the class distribution in the train, validation, and test datasets.
    """
    for name, df in zip(["Training", "Validation", "Test"], [train_df, val_df, test_df]):
        counts = df['label'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.bar(['Normal', 'Pneumonia'], [counts.get(0, 0), counts.get(1, 0)])
        plt.title(f"Class Distribution - {name} Data")
        plt.ylabel("Count")
        plt.show()


def visualize_balanced_predictions(model, loader, class_names, device, num_normal=10, num_pneumonia=10):
    """
    Visualizes balanced prediction images with Normal and Pneumonia labels.
    Highlights incorrect predictions in red.
    """
    model.eval()
    normal_images = []
    pneumonia_images = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for i in range(images.size(0)):
                # Convert the image tensor to a NumPy array for visualization
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                if image.shape[-1] == 1:  # Grayscale image
                    image = image.squeeze(-1)

                true_label = labels[i].item()
                pred_label = predictions[i].item()
                label_dict = {
                    "image": image,
                    "true_label": class_names[true_label],
                    "pred_label": class_names[pred_label],
                    "correct": true_label == pred_label,
                }

                # Collect balanced images
                if true_label == 0 and len(normal_images) < num_normal:
                    normal_images.append(label_dict)
                elif true_label == 1 and len(pneumonia_images) < num_pneumonia:
                    pneumonia_images.append(label_dict)

            # Stop if there are enough images
            if len(normal_images) >= num_normal and len(pneumonia_images) >= num_pneumonia:
                break

    # Combine images for visualization
    images_to_display = normal_images + pneumonia_images
    rows, cols = 4, 5  # Grid size
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))

    for idx, ax in enumerate(axes.flat):
        if idx < len(images_to_display):
            img_dict = images_to_display[idx]
            image = img_dict["image"]
            true_label = img_dict["true_label"]
            pred_label = img_dict["pred_label"]
            correct = img_dict["correct"]

            # Title with green if correct, red if incorrect
            title_color = "green" if correct else "red"
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
            ax.axis("off")
        else:
            ax.axis("off")  # Empty cells

    plt.tight_layout()
    plt.show()

def visualize_feature_maps_across_layers(model, input_image, device, num_cols=8, max_features=8):

    model.eval()
    input_image = input_image.to(device)

    feature_maps = []
    layer_names = []

    def hook_fn(module, _, output):
        feature_maps.append(output)
        layer_names.append(module.__class__.__name__)

    # Registrar hooks en todas las capas
    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook_fn))

    # Pasar la imagen a través del modelo
    with torch.no_grad():
        _ = model(input_image)

    # Eliminar los hooks después del forward pass
    for hook in hooks:
        hook.remove()

    # Visualizar los mapas de características para cada capa
    for idx, fmap in enumerate(feature_maps):
        fmap = fmap.squeeze(0)  # Elimina la dimensión de batch

        # Validar la forma del mapa de características
        if fmap.ndim < 3:
            print(f"Skipping layer {layer_names[idx]}: Invalid shape {fmap.shape}")
            continue

        num_features = fmap.shape[0]
        num_features = min(num_features, max_features)
        num_rows = (num_features + num_cols - 1) // num_cols

        plt.figure(figsize=(num_cols * 2, num_rows * 2))

        for i in range(num_features):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(fmap[i].cpu().numpy(), cmap='viridis')
            plt.axis('off')

        plt.suptitle(f"Feature Maps - Layer: {layer_names[idx]}")
        plt.tight_layout()
        plt.show()

def plot_roc_curve(y_true, y_pred, title="ROC Curve"):
    """
    Plots the ROC curve and calculates the AUC-ROC score.

    Args:
        y_true (list or numpy array): True binary labels.
        y_pred (list or numpy array): Predicted probabilities or scores.
        title (str): Title of the ROC plot.

    Returns:
        float: The AUC-ROC score.
    """
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC-ROC: {auc_roc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.show()

    return auc_roc

