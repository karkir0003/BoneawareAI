import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    num_epochs,
    device="cuda",
):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(
                enumerate(loader), total=len(loader), desc=f"{phase} Progress"
            )

            for i, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    try:
                        with autocast(
                            enabled=torch.cuda.is_available()
                        ):  # Mixed precision
                            outputs = model(inputs).view(
                                -1
                            )  # Ensure outputs have shape [batch_size]
                            labels = labels.view(
                                -1
                            )  # Ensure labels have shape [batch_size]
                            loss = criterion(outputs, labels)

                        if phase == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        # Convert logits to predictions
                        preds = (outputs > 0.5).float()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels)

                        progress_bar.set_postfix(
                            loss=loss.item(),
                            accuracy=(
                                running_corrects.double() / ((i + 1) * inputs.size(0))
                            ).item(),
                        )

                    except Exception as e:
                        print(f"Error in batch {i} of {phase}: {e}")
                        continue

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            train_history[f"{phase}_loss"].append(epoch_loss)
            train_history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, train_history


def train_model_kappa(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    num_epochs,
    device="cuda",
    body_part=None,
):
    best_model_wts = model.state_dict()
    best_cohen_kappa = 0.0  # Track the best Cohen Kappa score
    train_history = {
        "train_loss": [],
        "train_acc": [],
        "train_kappa": [],  # Add train Cohen's Kappa
        "valid_loss": [],
        "valid_acc": [],
        "valid_kappa": [],  # Add valid Cohen's Kappa
    }

    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = valid_loader

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            progress_bar = tqdm(
                enumerate(loader), total=len(loader), desc=f"{phase} Progress"
            )

            for i, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    try:
                        # with autocast(enabled=torch.cuda.is_available()):  # Mixed precision
                        # Mixed precision
                        with torch.amp.autocast(
                            device_type=device, enabled=torch.cuda.is_available()
                        ):
                            # Ensure outputs have shape [batch_size]
                            outputs = model(inputs).view(-1)
                            # Ensure labels have shape [batch_size]
                            labels = labels.view(-1)
                            loss = criterion(outputs, labels)

                        if phase == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        # Convert logits to predictions
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        progress_bar.set_postfix(
                            loss=loss.item(),
                            accuracy=(
                                running_corrects.double() / ((i + 1) * inputs.size(0))
                            ).item(),
                        )

                    except Exception as e:
                        print(f"Error in batch {i} of {phase}: {e}")
                        continue

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            # Calculate Cohen's Kappa for the current phase
            kappa = cohen_kappa_score(all_labels, all_preds)

            train_history[f"{phase}_loss"].append(epoch_loss)
            train_history[f"{phase}_acc"].append(epoch_acc.item())
            train_history[f"{phase}_kappa"].append(kappa)

            print(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Cohen's Kappa: {kappa:.4f}"
            )

            # Save model based on the best Cohen Kappa during validation
            if phase == "valid" and kappa > best_cohen_kappa:
                best_cohen_kappa = kappa
                best_model_wts = model.state_dict()
                if body_part is None:
                    torch.save(
                        model.state_dict(), f"best_model_kappa_epoch_{epoch+1}.pth"
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        f"best_model_{body_part}_kappa_epoch_{epoch+1}.pth",
                    )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    print(f"Best Validation Cohen's Kappa: {best_cohen_kappa:.4f}")
    model.load_state_dict(best_model_wts)
    return model, train_history
