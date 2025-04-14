import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def trainer(train_loader, val_loader, model, device):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    plt_step, plt_loss_train_record, plt_loss_valid_record, plt_acc_train_record, plt_acc_valid_record = [], [], [], [], []
    step = 0
    stale = 0
    epochs = 200
    best_acc = 0.0
    for epoch in range(epochs):
        train_acc, train_loss, val_acc, val_loss = 0, 0, 0, 0
        model.train()  # set the model to training mode
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for data, labels in train_pbar:
            data = data.float().to(device)
            labels = labels.to(device)
            # set grad to zero
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1

            # Return to its classification
            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred == labels).float().mean().item()
            train_loss += loss.item()

            # Setting Progress Bar Information
            train_pbar.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            train_pbar.set_postfix({'train_loss': loss.detach().item()})

        loss_epoch_train = train_loss / len(train_loader)
        plt_loss_train_record.append(loss_epoch_train)
        acc_epoch_train = train_acc / len(train_loader)
        plt_acc_train_record.append(acc_epoch_train)
        plt_step.append(epoch)

        # eval()
        model.eval()  # set the model to evaluation mode
        val_pbar = tqdm(val_loader, position=0, leave=True)
        with torch.no_grad():
            for data, labels in val_pbar:
                data = data.float().to(device)
                labels = labels.to(device)
                outputs = model(data)

                loss = criterion(outputs, labels)
                # loss_record.append(loss.detach().item())
                _, val_pred = torch.max(outputs, 1)
                val_acc += (
                            val_pred == labels).float().mean().item()  # get the index of the class with the highest probability
                # print(val_acc)
                val_loss += loss.item()
                val_pbar.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                val_pbar.set_postfix({'val_loss': loss.detach().item()})

            loss_epoch_valid = val_loss / len(valid_loader)
            # print("loss_epoch_valid:",loss_epoch_valid)
            plt_loss_valid_record.append(loss_epoch_valid)
            acc_epoch_valid = val_acc / len(valid_loader)
            plt_acc_valid_record.append(acc_epoch_valid)

            scheduler.step(acc_epoch_valid)
            # if the model improves, save a checkpoint at this epoch
            if acc_epoch_valid > best_acc:
                best_acc = acc_epoch_valid
                torch.save(model.state_dict(), 'saved_model.pkl')  # Save your best model
                print('saving model with acc {:.3f}'.format(best_acc))
                stale = 0
            else:
                stale += 1
                if stale > 20:
                    print(f"No improvment, early stopping")
                    break

        # scheduler.step()
    return plt_step, plt_loss_train_record, plt_loss_valid_record, plt_acc_train_record, plt_acc_valid_record

# 开始训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info]: Use {device} now!")
# KIR Dataset
model = ConvLF_Net(channel_in = 13,num_classes = 6).to(device)
# CRWU dataset
# model = ConvLF_Net(channel_in = 1,num_classes = 4).to(device)

step, plt_train_loss, plt_valid_loss, plt_acc_train, plt_acc_valid = trainer(train_loader, valid_loader, model, device)
