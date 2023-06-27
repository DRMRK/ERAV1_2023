import matplotlib.pyplot as plt


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def plot_losses_accuracies(
    train_acc,
    train_losses,
    test_acc,
    test_losses,
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def return_dataset_images(train_loader, total_images):
    """Shows images from dataloader 

    Args:
        train_loader (_type_): _description_
        total_images (_type_): _description_
    """
    batch_data, batch_label = next(iter(train_loader))
    
    fig = plt.figure()

    for i in range(total_images):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])