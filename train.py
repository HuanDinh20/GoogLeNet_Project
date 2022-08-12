import torch


def train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch_idx):
    """
    1. get inputs, labels. move to device
    2. zeros gradients
    3. perform inference
    4. Calculate loss
    5. backward
    6. optimizer steps -> adjust learning weights
    7. report
    """
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # 1.
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 3. zeros grad
        optimizer.zero_grad()
        # 4. perform inference
        outputs = model(inputs)
        # 5. calculate loss
        loss = loss_fn(outputs, labels)
        # 6. backward
        loss.backward()
        # 7. optimizer steps
        optimizer.step()

        running_loss += loss.item()

        if not(i % 1000):
            last_loss = running_loss / 1000
            print(f" Batch {i} Training Loss {last_loss}")
            x_tb = epoch_idx * len(train_loader) + i + 1
            summary_writer.add_scalar('Traing/Loss', last_loss, x_tb)
            running_loss = 0.0
    return last_loss


def per_epoch_activity(train_loader, val_loader, device, optimizer, model, loss_fn, summary_writer, epochs, timestamp):
    """
    at each epoch:
    1. perfrom validation
    2. saved the best model
    """
    best_val_loss = 0.0
    for epoch in range(epochs):

        model.train(True)
        avg_loss = train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch)
        model.train(False)

        running_val_loss = 0.0

        for i, data in enumerate(val_loader):
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels)
            running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        print(f"EPOCH {epoch} Training Loss {avg_loss} Validation Loss {avg_val_loss}")

        summary_writer.add_scalars('Training vs. Validation Loss',
                                   {'Training': avg_loss, 'Validation': avg_val_loss},
                                   epoch + 1)

        summary_writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = rf"saved_model\state_dict\GoogLeNet{timestamp}"
            torch.save(model.state_dict(), model_path)

